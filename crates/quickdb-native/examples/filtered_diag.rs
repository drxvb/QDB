//! Frozen small-index diagnostic of lazy filtered search (forecast F46).
//!
//! `cargo run --release -p quickdb-native --example filtered_diag`
//!
//! Builds 20,000 random 64-dimensional vectors (fixed seed, L2, m 16, ef_construction 200) and a
//! row per id with an indexed integer `bucket = id % 10`, then for 200 fixed queries compares
//! `knn_filtered` at ef 64, k 10 with exact filtered search. Three filters: `bucket` in `0..=0`
//! (10%), in `0..=4` (50%), and `Filter::All`, plus the 10% filter at ef 10. The search path
//! each filter takes is predicted from the
//! thresholds in `NativeStore::knn_filtered` (`src/lib.rs`): lazy when the sampled selectivity is
//! at least `max(sqrt(15 ef / N), 0.02)`, exact when at most `max(2 sqrt(ef N), 1024)` rows match.
//! At ef 64 and N 20,000 the lazy threshold is 0.219, so the 10% filter takes the exact path; the
//! lazy path is reached by the 50% filter, `Filter::All`, and the 10% filter at ef 10.
//! The store lives in `$QUICKDB_DATA` (or the temp directory) and is removed at the end.
use quickdb_api::conformance::{exact_distance, Lcg};
use quickdb_api::{
    ColumnDef, ColumnType, Filter, Hit, Metric, OpenOptions, Row, Store, TableDef, Value,
    VectorIndexDef,
};
use quickdb_native::NativeStore;
use std::collections::HashSet;
use std::path::PathBuf;

const N: u64 = 20_000;
const DIM: usize = 64;
const QUERIES: usize = 200;
const K: usize = 10;
const EF: usize = 64;
const SEED: u64 = 46;

type BoxError = Box<dyn std::error::Error>;

fn exact(base: &[Vec<f32>], q: &[f32], allowed: impl Fn(u64) -> bool) -> Vec<(u64, f32)> {
    let mut d: Vec<(u64, f32)> = (0..N)
        .filter(|&id| allowed(id))
        .map(|id| (id, exact_distance(Metric::L2, q, &base[id as usize])))
        .collect();
    d.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    d.truncate(K);
    d
}

fn fnv(acc: u64, id: u64) -> u64 {
    (acc ^ id).wrapping_mul(0x0100_0000_01b3)
}

struct Tally {
    lists_differ: usize,
    top1_differ: usize,
    missing_ids: usize,
    found: usize,
    digest: u64,
}

fn compare(got: &[Hit], want: &[(u64, f32)], t: &mut Tally) {
    let got_ids: Vec<u64> = got.iter().map(|h| h.id).collect();
    let want_ids: Vec<u64> = want.iter().map(|w| w.0).collect();
    if got_ids != want_ids {
        t.lists_differ += 1;
    }
    if got_ids.first() != want_ids.first() {
        t.top1_differ += 1;
    }
    let have: HashSet<u64> = got_ids.iter().copied().collect();
    let hit = want_ids.iter().filter(|id| have.contains(id)).count();
    t.found += hit;
    t.missing_ids += want_ids.len() - hit;
    t.digest = got_ids.iter().fold(t.digest, |a, &id| fnv(a, id));
}

fn main() -> Result<(), BoxError> {
    let root = std::env::var_os("QUICKDB_DATA").map_or_else(std::env::temp_dir, PathBuf::from);
    std::fs::create_dir_all(&root)?;
    let dir = tempfile::tempdir_in(&root)?;
    let mut rng = Lcg(SEED);
    let base: Vec<Vec<f32>> = (0..N).map(|_| rng.vector(DIM)).collect();
    let mut qrng = Lcg(SEED + 1);
    let queries: Vec<Vec<f32>> = (0..QUERIES).map(|_| qrng.vector(DIM)).collect();

    let mut s = NativeStore::open(dir.path(), &OpenOptions::default())?;
    s.define_table(&TableDef {
        name: "diag".into(),
        columns: vec![ColumnDef {
            name: "bucket".into(),
            ty: ColumnType::Int,
            indexed: true,
        }],
    })?;
    s.define_vector_index(&VectorIndexDef {
        name: "emb".into(),
        dim: DIM,
        metric: Metric::L2,
        m: 16,
        ef_construction: 200,
    })?;
    let rows: Vec<Row> = (0..N)
        .map(|id| Row {
            id,
            values: vec![Value::Int((id % 10) as i64)],
        })
        .collect();
    s.upsert_rows("diag", &rows)?;
    let items: Vec<(u64, &[f32])> = base
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v.as_slice()))
        .collect();
    for chunk in items.chunks(5_000) {
        s.upsert_vectors("emb", chunk)?;
    }
    s.flush()?;
    println!(
        "DIAG_INDEX n={N} dim={DIM} metric=l2 m=16 ef_construction=200 seed={SEED} queries={QUERIES} k={K} ef={EF}"
    );

    let lazy_from = |ef: usize| (15.0 * ef as f64 / N as f64).sqrt().max(0.02);
    let exact_up_to = |ef: usize| (2.0 * (ef as f64 * N as f64).sqrt()).max(1024.0) as u64;
    // The library estimates selectivity from 256 evenly spread positions; ids were inserted in
    // order into a fresh index, so position n holds id n.
    let sampled = |tenths: u64| {
        let hits = (0..256u64).filter(|i| (i * N / 256) % 10 < tenths).count();
        hits as f64 / 256.0
    };
    println!(
        "DIAG_THRESHOLDS ef={EF} lazy_from_selectivity={:.3} exact_up_to_rows={}",
        lazy_from(EF),
        exact_up_to(EF)
    );

    let range = |max: i64| Filter::IntRange {
        column: "bucket".into(),
        min: 0,
        max,
    };
    // (name, filter, matching tenths of the ids, ef). The last case is supplementary: at ef 64 a
    // 10% filter on 20,000 vectors is below the lazy threshold, at ef 10 it is above it.
    let cases: [(&str, Option<Filter>, u64, usize); 5] = [
        ("knn_unfiltered", None, 10, EF),
        ("bucket_eq_0_10pct", Some(range(0)), 1, EF),
        ("bucket_le_4_50pct", Some(range(4)), 5, EF),
        ("filter_all", Some(Filter::All), 10, EF),
        ("bucket_eq_0_10pct_ef10", Some(range(0)), 1, 10),
    ];
    let mut lazy_lists = 0;
    let mut lazy_top1 = 0;
    for (name, filter, tenths, ef) in &cases {
        let allowed_rows = N * tenths / 10;
        let estimate = sampled(*tenths);
        let path = match filter {
            None => "graph",
            Some(_) if estimate >= lazy_from(*ef) => "lazy",
            Some(_) if allowed_rows <= exact_up_to(*ef) => "exact",
            Some(_) => "mask",
        };
        let mut t = Tally {
            lists_differ: 0,
            top1_differ: 0,
            missing_ids: 0,
            found: 0,
            digest: 0xcbf2_9ce4_8422_2325,
        };
        for q in &queries {
            let got = match filter {
                None => s.knn("emb", q, K, *ef)?,
                Some(f) => s.knn_filtered("emb", q, K, *ef, "diag", f)?,
            };
            let want = exact(&base, q, |id| id % 10 < *tenths);
            compare(&got, &want, &mut t);
        }
        if path == "lazy" && *ef == EF {
            lazy_lists += t.lists_differ;
            lazy_top1 += t.top1_differ;
        }
        println!(
            "DIAG case={name} ef={ef} allowed_rows={allowed_rows} sampled_selectivity={estimate:.3} predicted_path={path} lists_differ={}/{QUERIES} top1_differ={}/{QUERIES} missing_ids={} recall@10={:.4} digest={:016x}",
            t.lists_differ,
            t.top1_differ,
            t.missing_ids,
            t.found as f64 / (QUERIES * K) as f64,
            t.digest
        );
    }
    // F46: at least one nearest id differs between lazy filtered and exact filtered search.
    let verdict = if lazy_lists > 0 { "held" } else { "wrong" };
    println!(
        "DIAG_F46 ef={EF} lazy_queries_with_a_different_list={lazy_lists} lazy_top1_differ={lazy_top1} verdict={verdict}"
    );
    drop(s);
    dir.close()?;
    Ok(())
}
