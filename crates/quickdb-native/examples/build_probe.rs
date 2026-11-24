use quickdb_api::{
    ColumnDef, ColumnType, Filter, Metric, OpenOptions, Row, Store, TableDef, Value, VectorIndexDef,
};
use quickdb_native::NativeStore;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

fn records(path: &str, n: usize) -> Result<Vec<Vec<u32>>, Box<dyn std::error::Error>> {
    let mut input = BufReader::new(File::open(path)?);
    let mut result = Vec::with_capacity(n);
    for _ in 0..n {
        let mut word = [0; 4];
        input.read_exact(&mut word)?;
        let dim = u32::from_le_bytes(word) as usize;
        if !(1..=4096).contains(&dim) {
            return Err("invalid record dimension".into());
        }
        let mut row = Vec::with_capacity(dim);
        for _ in 0..dim {
            input.read_exact(&mut word)?;
            row.push(u32::from_le_bytes(word));
        }
        result.push(row);
    }
    Ok(result)
}

fn floats(path: &str, n: usize) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    Ok(records(path, n)?
        .into_iter()
        .map(|r| r.into_iter().map(f32::from_bits).collect())
        .collect())
}

#[allow(dead_code)]
#[path = "../src/snapshot.rs"]
mod snapshot;

fn graph_digest(path: &Path) -> Result<u64, Box<dyn std::error::Error>> {
    let mut input = BufReader::new(File::open(path.join("emb.hnsw"))?);
    let payload = input
        .get_ref()
        .metadata()?
        .len()
        .checked_sub(4)
        .ok_or("short snapshot")?;
    let (_, graph) = snapshot::decode(&mut input, payload)?;
    let mut bytes = Vec::new();
    snapshot::encode_digest(&mut bytes, &graph.parts())?;
    Ok(bytes
        .into_iter()
        .fold(14695981039346656037u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(1099511628211)
        }))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "200000".into())
        .parse()?;
    if !(10..=1_000_000).contains(&n) {
        return Err("N must be in 10..=1000000".into());
    }
    let dataset = std::env::var("QUICKDB_BUILD_DATASET").unwrap_or_else(|_| "sift".into());
    if dataset != "sift" && dataset != "gist" {
        return Err("dataset must be sift or gist".into());
    }
    let query_count: usize = std::env::var("QUICKDB_BUILD_QUERIES")
        .unwrap_or_else(|_| "1000".into())
        .parse()?;
    if query_count == 0 || query_count > 1000 {
        return Err("query count must be in 1..=1000".into());
    }
    let data = floats(&format!("/datasets/{dataset}/{dataset}_base.fvecs"), n)?;
    let queries = floats(
        &format!("/datasets/{dataset}/{dataset}_query.fvecs"),
        query_count,
    )?;
    let truth: Vec<Vec<u64>> = if n == 1_000_000 {
        records(
            &format!("/datasets/{dataset}/{dataset}_groundtruth.ivecs"),
            query_count,
        )?
        .into_iter()
        .map(|r| r.into_iter().take(10).map(|id| u64::from(id) + 1).collect())
        .collect()
    } else {
        let start = Instant::now();
        let truth = queries
            .iter()
            .map(|q| {
                let mut best: Vec<(f32, u64)> = Vec::with_capacity(11);
                for (i, v) in data.iter().enumerate() {
                    let distance: f32 = q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
                    let id = i as u64 + 1;
                    let pos =
                        best.partition_point(|&(d, j)| d < distance || (d == distance && j < id));
                    if pos < 10 {
                        best.insert(pos, (distance, id));
                        best.truncate(10);
                    }
                }
                best.into_iter().map(|(_, id)| id).collect()
            })
            .collect();
        println!(
            "EXACT_TRUTH n={n} seconds={:.6}",
            start.elapsed().as_secs_f64()
        );
        truth
    };
    let root = std::env::var("QUICKDB_DATA").unwrap_or_else(|_| "/data".into());
    let dir = tempfile::tempdir_in(Path::new(&root))?;
    let path = std::env::args()
        .nth(2)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| dir.path().to_path_buf());
    let existing = std::env::args().nth(3).as_deref() == Some("reopen");
    if !existing && path.exists() && std::fs::read_dir(&path)?.next().is_some() {
        return Err("build store must be empty".into());
    }
    let mut store = NativeStore::open(&path, &OpenOptions::default())?;
    if existing && store.index_rebuilds() != 0 {
        return Err("existing snapshot rebuilt".into());
    }
    if !existing {
        store.define_vector_index(&VectorIndexDef {
            name: "emb".into(),
            dim: data[0].len(),
            metric: Metric::L2,
            m: 16,
            ef_construction: 200,
        })?;
    }
    let start = Instant::now();
    if !existing {
        for (batch_no, batch) in data.chunks(1000).enumerate() {
            let items: Vec<_> = batch
                .iter()
                .enumerate()
                .map(|(i, v)| ((batch_no * 1000 + i + 1) as u64, v.as_slice()))
                .collect();
            store.upsert_vectors("emb", &items)?;
            if (batch_no + 1).is_multiple_of(100) {
                println!(
                    "BUILD_PROGRESS loaded={} seconds={:.6}",
                    (batch_no + 1) * 1000,
                    start.elapsed().as_secs_f64()
                );
            }
        }
    }
    let build = start.elapsed().as_secs_f64();
    println!("BUILD_TIME n={n} seconds={build:.6}");
    let mut times = Vec::with_capacity(1000);
    let mut found = 0;
    let mut digest = 0u64;
    for (q, wanted) in queries.iter().zip(&truth) {
        let start = Instant::now();
        let hits = store.knn("emb", q, 10, 64)?;
        times.push(start.elapsed().as_secs_f64() * 1e6);
        found += hits.iter().filter(|h| wanted.contains(&h.id)).count();
        for hit in hits {
            digest = digest
                .wrapping_mul(1099511628211)
                .wrapping_add(hit.id ^ u64::from(hit.distance.to_bits()));
        }
    }
    times.sort_by(f64::total_cmp);
    println!(
        "BUILD_PROBE n={n} build_seconds={build:.6} recall_at_10={:.6} median_search_us={:.6}",
        found as f64 / (query_count * 10) as f64,
        times[query_count / 2]
    );
    println!("SEARCH_DIGEST {digest:016x}");
    println!("BUILD_STORE path={} existing={existing}", path.display());
    if std::env::var_os("QUICKDB_BUILD_FILTER").is_some() {
        store.define_table(&TableDef {
            name: "docs".into(),
            columns: vec![ColumnDef {
                name: "group_id".into(),
                ty: ColumnType::Int,
                indexed: true,
            }],
        })?;
        for batch in (1..=n as u64).collect::<Vec<_>>().chunks(1000) {
            let rows: Vec<_> = batch
                .iter()
                .map(|&id| Row {
                    id,
                    values: vec![Value::Int((id % 2) as i64)],
                })
                .collect();
            store.upsert_rows("docs", &rows)?;
        }
        let filter = Filter::Eq {
            column: "group_id".into(),
            value: Value::Int(0),
        };
        let mut times = Vec::new();
        let mut digest = 0u64;
        for q in &queries {
            let start = Instant::now();
            let hits = store.knn_filtered("emb", q, 10, 64, "docs", &filter)?;
            times.push(start.elapsed().as_secs_f64() * 1e6);
            assert_eq!(hits.len(), 10);
            for hit in hits {
                digest = digest
                    .wrapping_mul(1099511628211)
                    .wrapping_add(hit.id ^ u64::from(hit.distance.to_bits()));
            }
        }
        times.sort_by(f64::total_cmp);
        println!(
            "FILTER_PROBE n={n} selectivity=0.5 median_search_us={:.6} digest={digest:016x}",
            times[499]
        );
    }
    let start = Instant::now();
    store.flush()?;
    println!("FLUSH seconds={:.6}", start.elapsed().as_secs_f64());
    println!("GRAPH_DIGEST {:016x}", graph_digest(&path)?);
    drop(store);
    let start = Instant::now();
    let store = NativeStore::open(&path, &OpenOptions::default())?;
    println!(
        "REOPEN seconds={:.6} rebuilds={}",
        start.elapsed().as_secs_f64(),
        store.index_rebuilds()
    );
    assert_eq!(store.count_vectors("emb")?, n as u64);
    Ok(())
}
