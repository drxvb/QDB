//! Process-isolated construction switches. Fixtures captured from committed original
//! hnsw.rs. QUICKDB_CAUSE_CAPTURE_BASELINE=1 with --nocapture explicitly bypasses
//! fixture assertions to print candidate baselines for coordinator verification.
use quickdb_api::{conformance::Lcg, Metric, OpenOptions, Store, VectorIndexDef};
use quickdb_native::NativeStore;
use std::collections::HashSet;
use std::io::{BufReader, Read};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

const CASES: [(usize, bool, Option<u64>); 4] = [
    (16, false, Some(0x09d074ac6ca2b87f)),
    (19, false, Some(0x7024e62947e16208)),
    (16, true, Some(0x10507b66c548f627)),
    (19, true, Some(0x613cc88fb05b51ab)),
];
const COUNT: usize = 768;
const QUERIES: usize = 64;
const K: usize = 10;
const RESULT: &str = "cause-result.txt";

#[allow(dead_code)]
#[path = "../src/snapshot.rs"]
mod snapshot;

// Excludes randomized store identity, vector-state nonce and HashMap iteration.
// Matches the graph-only digest in examples/build_probe.rs.
fn digest(path: &Path) -> u64 {
    let bytes = std::fs::read(path.join("emb.hnsw")).expect("snapshot");
    let payload = &bytes[..bytes.len().checked_sub(4).expect("checksum trailer")];
    let (_, g) =
        snapshot::decode(payload, payload.len() as u64).expect("unchanged snapshot layout");
    let mut bytes = Vec::new();
    snapshot::encode_digest(&mut bytes, &g.parts()).expect("serialize graph");
    bytes.into_iter().fold(14695981039346656037, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(1099511628211)
    })
}

fn vectors(seed: u64, count: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = Lcg(seed);
    (0..count).map(|_| rng.vector(dim)).collect()
}

// Independent f64 exact scan, not a second invocation of the HNSW kernel.
fn exact_distance(a: &[f32], b: &[f32], cosine: bool) -> f64 {
    if cosine {
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        let aa: f64 = a.iter().map(|&x| f64::from(x).powi(2)).sum();
        let bb: f64 = b.iter().map(|&x| f64::from(x).powi(2)).sum();
        1.0 - dot / (aa * bb).sqrt()
    } else {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum()
    }
}

fn recall(store: &NativeStore, data: &[Vec<f32>], queries: &[Vec<f32>], cosine: bool) -> f64 {
    let mut correct = 0;
    for q in queries {
        let mut truth: Vec<_> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (exact_distance(q, v, cosine), i as u64 + 1))
            .collect();
        truth.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let truth: HashSet<_> = truth.iter().take(K).map(|x| x.1).collect();
        let hits = store.knn("emb", q, K, 128).expect("knn");
        assert_eq!(hits.len(), K);
        let ids: HashSet<_> = hits.iter().map(|h| h.id).collect();
        assert_eq!(ids.len(), K, "duplicate hits");
        assert!(ids.iter().all(|&id| id > 0 && id <= COUNT as u64));
        for pair in hits.windows(2) {
            assert!(pair[0].distance <= pair[1].distance, "unordered distances");
            if pair[0].distance == pair[1].distance {
                assert!(pair[0].id < pair[1].id, "unordered distance ties");
            }
        }
        for hit in &hits {
            let exact = exact_distance(q, &data[hit.id as usize - 1], cosine);
            assert!(hit.distance.is_finite());
            assert!(
                (f64::from(hit.distance) - exact).abs() <= 1e-4 * exact.abs().max(1.0),
                "incorrect distance after kernel selection: {hit:?}, exact={exact}"
            );
        }
        correct += ids.intersection(&truth).count();
    }
    correct as f64 / (queries.len() * K) as f64
}

/// Can be run alone by the coordinator. Writes digest and recall to RESULT and
/// prints both even in capture mode. No process-global environment is mutated.
#[test]
fn build_cause_child() {
    let Some(path) = std::env::var_os("QUICKDB_CAUSE_TEST_DIR") else {
        return;
    };
    let path = Path::new(&path);
    let dim: usize = std::env::var("QUICKDB_CAUSE_TEST_DIM")
        .expect("dim")
        .parse()
        .expect("numeric dim");
    let cosine = std::env::var("QUICKDB_CAUSE_TEST_COSINE").expect("metric") == "1";
    let reopen = std::env::var("QUICKDB_CAUSE_TEST_REOPEN").expect("mode") == "1";
    let data = vectors(0x666c6f6174636175, COUNT, dim);
    // Separate seed and generated vectors: no indexed-vector self queries.
    let queries = vectors(0x7175657279636175, QUERIES, dim);
    let mut store = NativeStore::open(path, &OpenOptions::default()).expect("open");
    if reopen {
        assert_eq!(
            store.index_rebuilds(),
            0,
            "cross-kernel reopen rebuilt graph"
        );
    } else {
        store
            .define_vector_index(&VectorIndexDef {
                name: "emb".into(),
                dim,
                metric: if cosine { Metric::Cosine } else { Metric::L2 },
                m: 16,
                ef_construction: 200,
            })
            .expect("define index");
        for (batch, chunk) in data.chunks(257).enumerate() {
            let items: Vec<_> = chunk
                .iter()
                .enumerate()
                .map(|(i, v)| ((batch * 257 + i + 1) as u64, v.as_slice()))
                .collect();
            store.upsert_vectors("emb", &items).expect("insert");
        }
        store.flush().expect("snapshot");
    }
    let score = recall(&store, &data, &queries, cosine);
    assert!(score >= 0.95, "unexpectedly low recall: {score}");
    let graph = digest(path);
    let result = format!("{graph:016x} {score:.9}\n");
    println!("BUILD_CAUSE dim={dim} cosine={cosine} reopen={reopen} {result}");
    std::fs::write(path.join(RESULT), result).expect("write child result");
}

fn run(
    path: &Path,
    dim: usize,
    cosine: bool,
    margin: Option<&str>,
    fma: Option<&str>,
    reopen: bool,
) -> (u64, f64) {
    let mut cmd = Command::new(std::env::current_exe().expect("test binary"));
    cmd.args(["--exact", "build_cause_child", "--nocapture"])
        .env("QUICKDB_CAUSE_TEST_DIR", path)
        .env("QUICKDB_CAUSE_TEST_DIM", dim.to_string())
        .env("QUICKDB_CAUSE_TEST_COSINE", if cosine { "1" } else { "0" })
        .env("QUICKDB_CAUSE_TEST_REOPEN", if reopen { "1" } else { "0" })
        .env("QUICKDB_BUILD_WORKERS", "1")
        .env_remove("QUICKDB_BUILD_FLOAT_OFF")
        .env_remove("QUICKDB_BUILD_SCALAR")
        .env_remove("QUICKDB_BUILD_NO_PACKED")
        .env_remove("QUICKDB_BUILD_FRESH_VISITED")
        .env_remove("QUICKDB_BUILD_NO_PREFETCH")
        .env_remove("QUICKDB_PREFETCH_HINTS")
        .env_remove("QUICKDB_BUILD_PROFILE")
        .env_remove("QUICKDB_BUILD_MARGIN_PCT")
        .env_remove("QUICKDB_BUILD_FMA");
    if let Some(value) = margin {
        cmd.env("QUICKDB_BUILD_MARGIN_PCT", value);
    }
    if let Some(value) = fma {
        cmd.env("QUICKDB_BUILD_FMA", value);
    }
    assert!(cmd.status().expect("run child").success(), "child failed");
    let text = std::fs::read_to_string(path.join(RESULT)).expect("child result");
    let mut fields = text.split_whitespace();
    let graph = u64::from_str_radix(fields.next().expect("digest"), 16).expect("hex digest");
    let score = fields
        .next()
        .expect("recall")
        .parse()
        .expect("numeric recall");
    assert!(fields.next().is_none());
    (graph, score)
}

#[test]
fn switches_preserve_default_fixture_recall_and_cross_kernel_snapshots() {
    let root = std::env::var_os("QUICKDB_DATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let capture = std::env::var("QUICKDB_CAUSE_CAPTURE_BASELINE").as_deref() == Ok("1");
    for (dim, cosine, expected) in CASES {
        let baseline = tempfile::tempdir_in(&root).expect("baseline dir");
        let (original, default_recall) = run(baseline.path(), dim, cosine, None, None, false);
        println!("BUILD_CAUSE_FIXTURE dim={dim} cosine={cosine} digest=0x{original:016x} recall={default_recall:.9}");
        if capture {
            continue;
        }
        assert_eq!(
            Some(original),
            expected,
            "capture committed original hnsw.rs with QUICKDB_CAUSE_CAPTURE_BASELINE=1, then pin CASES"
        );
        for (margin, fma) in [
            (Some("0"), None),
            (None, Some("1")),
            (Some("0"), Some("1")),
            (Some("invalid"), Some("invalid")),
            (Some("12"), Some("0")),
        ] {
            let dir = tempfile::tempdir_in(&root).expect("switch dir");
            let (built, score) = run(dir.path(), dim, cosine, margin, fma, false);
            assert!(score + 0.01 + f64::EPSILON >= default_recall,
                "recall regression dim={dim} cosine={cosine} margin={margin:?} fma={fma:?}: {score} vs {default_recall}");
            if margin == Some("invalid") {
                assert_eq!(built, original, "invalid switch changed default graph");
            }
            let snapshot = std::fs::read(dir.path().join("emb.hnsw")).expect("snapshot bytes");
            // Both directions: FMA-built -> default and default-built -> FMA.
            let opposite = if fma == Some("1") { None } else { Some("1") };
            let (reopened, reopened_recall) = run(dir.path(), dim, cosine, None, opposite, true);
            assert_eq!(built, reopened, "reopen changed graph");
            assert!(reopened_recall + 0.01 + f64::EPSILON >= default_recall);
            assert_eq!(
                snapshot,
                std::fs::read(dir.path().join("emb.hnsw")).expect("unchanged snapshot")
            );
        }
        let (reopened, score) = run(baseline.path(), dim, cosine, None, Some("1"), true);
        assert_eq!(reopened, original);
        assert!(score + 0.01 + f64::EPSILON >= default_recall);
    }
}

fn read_word(input: &mut impl Read) -> u32 {
    let mut word = [0; 4];
    input.read_exact(&mut word).expect("complete vector word");
    u32::from_le_bytes(word)
}

fn read_fvecs(path: &Path, count: usize) -> Vec<Vec<f32>> {
    let mut input = BufReader::new(std::fs::File::open(path).expect("fvecs file"));
    let mut result = Vec::with_capacity(count);
    let mut dimension = None;
    for _ in 0..count {
        let dim = read_word(&mut input) as usize;
        assert!((1..=65536).contains(&dim), "invalid fvecs dimension");
        assert_eq!(*dimension.get_or_insert(dim), dim, "inconsistent fvecs");
        let vector: Vec<_> = (0..dim)
            .map(|_| f32::from_bits(read_word(&mut input)))
            .collect();
        assert!(vector.iter().all(|x| x.is_finite()), "nonfinite fvecs");
        result.push(vector);
    }
    result
}

fn read_truth(path: &Path, count: usize, n: usize) -> Vec<HashSet<u64>> {
    let mut input = BufReader::new(std::fs::File::open(path).expect("ivecs truth file"));
    (0..count)
        .map(|_| {
            let width = read_word(&mut input) as usize;
            assert!((K..=n).contains(&width), "invalid truth width");
            let mut ids = HashSet::with_capacity(K);
            for rank in 0..width {
                let id = read_word(&mut input) as usize;
                assert!(id < n, "truth ID outside exact prefix");
                if rank < K {
                    assert!(ids.insert(id as u64 + 1), "duplicate exact truth ID");
                }
            }
            ids
        })
        .collect()
}

/// Narrow benchmark entry point, inert during ordinary workspace tests.
/// Invoke the test binary with --exact gist_cause_probe --nocapture. The harness
/// supplies CAUSE_BASE/QUERY/TRUTH/N/QUERIES/DB/RESULT. Truth is exhaustive for
/// this exact prefix, with zero-based IDs. Switches and workers come from the
/// harness environment. BUILD_PROFILE stderr remains the source of graph time.
#[test]
fn gist_cause_probe() {
    let Some(base) = std::env::var_os("CAUSE_BASE") else {
        return;
    };
    let n: usize = std::env::var("CAUSE_N")
        .expect("CAUSE_N")
        .parse()
        .expect("numeric prefix size");
    let nq: usize = std::env::var("CAUSE_QUERIES")
        .expect("CAUSE_QUERIES")
        .parse()
        .expect("numeric query count");
    assert!((K..=200000).contains(&n));
    assert!((1..=1000).contains(&nq));
    let query = std::env::var_os("CAUSE_QUERY").expect("CAUSE_QUERY");
    let truth = std::env::var_os("CAUSE_TRUTH").expect("CAUSE_TRUTH");
    let db = std::env::var_os("CAUSE_DB").expect("CAUSE_DB");
    let result = std::env::var_os("CAUSE_RESULT").expect("CAUSE_RESULT");
    let started = Instant::now();
    let data = read_fvecs(Path::new(&base), n);
    let queries = read_fvecs(Path::new(&query), nq);
    let truth = read_truth(Path::new(&truth), nq, n);
    let dim = data[0].len();
    assert!(queries.iter().all(|q| q.len() == dim));
    let load_s = started.elapsed().as_secs_f64();
    let path = Path::new(&db);
    assert!(
        !path.join("quickdb.sqlite").exists(),
        "probe requires fresh store"
    );
    let started = Instant::now();
    let mut store = NativeStore::open(path, &OpenOptions::default()).expect("probe open");
    store
        .define_vector_index(&VectorIndexDef {
            name: "emb".into(),
            dim,
            metric: Metric::L2,
            m: 16,
            ef_construction: 200,
        })
        .expect("probe index");
    let open_s = started.elapsed().as_secs_f64();
    let started = Instant::now();
    // Exactly one profile record per batch, matching the campaign parser.
    for (batch, chunk) in data.chunks(1000).enumerate() {
        let items: Vec<_> = chunk
            .iter()
            .enumerate()
            .map(|(i, v)| ((batch * 1000 + i + 1) as u64, v.as_slice()))
            .collect();
        store.upsert_vectors("emb", &items).expect("probe batch");
    }
    let build_s = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let mut recalls = [0.0; 2];
    for (slot, ef) in [64, 256].into_iter().enumerate() {
        let mut correct = 0;
        for (q, expected) in queries.iter().zip(&truth) {
            let hits = store.knn("emb", q, K, ef).expect("probe knn");
            assert_eq!(hits.len(), K);
            let mut seen = HashSet::with_capacity(K);
            for hit in &hits {
                assert!((1..=n as u64).contains(&hit.id));
                assert!(seen.insert(hit.id), "duplicate probe result");
                assert!(hit.distance.is_finite());
                correct += usize::from(expected.contains(&hit.id));
            }
            for pair in hits.windows(2) {
                assert!(pair[0].distance <= pair[1].distance);
            }
        }
        recalls[slot] = correct as f64 / (nq * K) as f64;
        println!(
            "CAUSE_PROBE n={n} queries={nq} ef={ef} recall={:.9}",
            recalls[slot]
        );
    }
    let query_s = started.elapsed().as_secs_f64();
    let report = format!(
        "{{\"dataset\":{{\"base_count\":{n}}},\"vectors\":{n},\"queries\":{{\"knn_ef64\":{{\"samples\":{nq},\"recall_at_10\":{}}},\"knn_ef256\":{{\"samples\":{nq},\"recall_at_10\":{}}}}},\"phases_seconds\":{{\"load\":{load_s},\"open\":{open_s},\"vector_upsert\":{build_s},\"knn\":{query_s}}}}}\n",
        recalls[0], recalls[1]
    );
    std::fs::write(Path::new(&result), report).expect("write probe JSON");
}
