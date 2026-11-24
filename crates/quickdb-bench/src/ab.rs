//! Paired, order-balanced query measurements on one immutable persisted graph.
use crate::{exact_top10_metric, Dataset};
use quickdb_api::{Error, OpenOptions, Result, Store, VectorIndexDef};
use serde::Serialize;
use std::path::Path;
use std::time::Instant;

#[cfg(feature = "native-counter")]
extern "C" {
    fn quickdb_distance_count(action: u8) -> u64;
}
fn counter(reset: bool) -> Option<u64> {
    #[cfg(feature = "native-counter")]
    {
        // The feature is linked only with native qbench. The exported function takes
        // a plain integer, accesses thread-local state, and has no pointer arguments.
        Some(unsafe { quickdb_distance_count(u8::from(reset)) })
    }
    #[cfg(not(feature = "native-counter"))]
    {
        let _ = reset;
        None
    }
}
fn bad(s: &str) -> Error {
    Error::Schema(s.into())
}
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}
fn interval(pairs: &[[f64; 2]]) -> [f64; 2] {
    let mut rng = quickdb_api::conformance::Lcg(0x41424241);
    let mut ratios = Vec::with_capacity(10000);
    for _ in 0..10000 {
        let sampled: Vec<_> = (0..pairs.len())
            .map(|_| pairs[rng.next_u32() as usize % pairs.len()])
            .collect();
        let a: Vec<_> = sampled.iter().map(|p| p[0]).collect();
        let b: Vec<_> = sampled.iter().map(|p| p[1]).collect();
        ratios.push(median(&a) / median(&b));
    }
    ratios.sort_by(f64::total_cmp);
    [ratios[249], ratios[9749]]
}
fn load() -> Result<String> {
    // Only Linux has this file; elsewhere the report says so instead of failing the run.
    match std::fs::read_to_string("/proc/loadavg") {
        Ok(text) => Ok(text.trim().into()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok("unavailable".into()),
        Err(e) => Err(e.into()),
    }
}
fn configure(old: bool) {
    if old {
        std::env::set_var("QUICKDB_BUILD_SCALAR", "1");
        std::env::set_var("QUICKDB_BUILD_NO_PREFETCH", "1");
        std::env::set_var("QUICKDB_PREFETCH_HINTS", "0");
    } else {
        std::env::remove_var("QUICKDB_BUILD_SCALAR");
        std::env::remove_var("QUICKDB_BUILD_NO_PREFETCH");
        std::env::set_var("QUICKDB_PREFETCH_HINTS", "4");
    }
}
struct EnvGuard(Vec<(&'static str, Option<std::ffi::OsString>)>);
impl EnvGuard {
    fn new() -> Self {
        Self(
            [
                "QUICKDB_BUILD_SCALAR",
                "QUICKDB_PREFETCH_HINTS",
                "QUICKDB_BUILD_NO_PREFETCH",
            ]
            .into_iter()
            .map(|key| (key, std::env::var_os(key)))
            .collect(),
        )
    }
}
impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in &self.0 {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
}
#[derive(Serialize)]
struct Block {
    pair: usize,
    arm: usize,
    order: usize,
    median_us: f64,
    elapsed_seconds: f64,
    correct: usize,
    distances: Option<u64>,
    fingerprint: u64,
    load_before: String,
    load_after: String,
}
/// Run at least ten paired blocks per trial. Each block contains all supplied queries.
/// Environment switches are process-global: call only from a single-threaded CLI.
pub fn run<S: Store>(
    data: &Dataset,
    path: &Path,
    pairs: usize,
    trials: usize,
    old_b: bool,
    ef: usize,
) -> Result<serde_json::Value> {
    if pairs < 10 || trials == 0 || ef == 0 {
        return Err(bad("ab needs pairs >= 10, trials > 0 and ef > 0"));
    }
    let queries = data
        .query_vectors
        .as_ref()
        .ok_or_else(|| bad("ab requires ANN queries"))?;
    if queries.is_empty() || data.vectors.is_empty() {
        return Err(bad("ab requires nonempty vectors and queries"));
    }
    if path.exists() && std::fs::read_dir(path)?.next().is_some() {
        return Err(bad("ab data-dir must be empty"));
    }
    let _guard = EnvGuard::new();
    configure(false);
    let prepare = Instant::now();
    let truth = match &data.truth {
        Some(truth) => truth.clone(),
        None => queries
            .iter()
            .map(|q| exact_top10_metric(&data.vectors, q, None, data.metric))
            .collect(),
    };
    if truth.len() != queries.len() || truth.iter().any(Vec::is_empty) {
        return Err(bad("ab truth must contain neighbors for every query"));
    }
    let wanted: usize = truth.iter().map(Vec::len).sum();
    let truth_seconds = prepare.elapsed().as_secs_f64();
    println!(
        "AB_PREPARED truth_seconds={truth_seconds:.3} queries={}",
        queries.len()
    );
    let build = Instant::now();
    let mut store = S::open(path, &OpenOptions::default())?;
    store.define_vector_index(&VectorIndexDef {
        name: "emb".into(),
        dim: data.dim,
        metric: data.metric,
        m: 16,
        ef_construction: 200,
    })?;
    let items: Vec<_> = data
        .vectors
        .iter()
        .map(|(id, v)| (*id, v.as_slice()))
        .collect();
    store.upsert_vectors("emb", &items)?;
    store.flush()?;
    drop(store);
    let build_seconds = build.elapsed().as_secs_f64();
    println!("AB_BUILT seconds={build_seconds:.3}");
    let mut reports = Vec::new();
    for trial in 0..trials {
        let before = load()?;
        let mut blocks = Vec::new();
        let mut paired = Vec::new();
        for pair in 0..pairs {
            let mut times = [0.0; 2];
            // Alternating AB / BA balances which arm runs first. Never discard a block.
            for order in 0..2 {
                let arm = (order + pair + trial) % 2;
                configure(old_b && arm == 1);
                let store = S::open(
                    path,
                    &OpenOptions {
                        create: false,
                        sync_each_write: false,
                    },
                )?;
                // Equal full-query warmup removes cold-open/cache effects from either arm.
                for q in queries {
                    std::hint::black_box(store.knn("emb", q, 10, ef)?);
                }
                counter(true);
                let load_before = load()?;
                let start = Instant::now();
                let mut latencies = Vec::with_capacity(queries.len());
                let mut results = Vec::with_capacity(queries.len());
                for q in queries {
                    let query_start = Instant::now();
                    let hits = std::hint::black_box(store.knn("emb", q, 10, ef)?);
                    latencies.push(query_start.elapsed().as_secs_f64() * 1e6);
                    results.push(hits);
                }
                let elapsed_seconds = start.elapsed().as_secs_f64();
                let distances = counter(false).filter(|count| *count > 0);
                let load_after = load()?;
                let mut fingerprint = 0xcbf29ce484222325u64;
                let mut correct = 0;
                for (hits, truth) in results.iter().zip(&truth) {
                    correct += truth
                        .iter()
                        .filter(|id| hits.iter().any(|h| h.id == **id))
                        .count();
                    for h in hits {
                        fingerprint = (fingerprint ^ h.id).wrapping_mul(0x100000001b3);
                        fingerprint = (fingerprint ^ u64::from(h.distance.to_bits()))
                            .wrapping_mul(0x100000001b3);
                    }
                }
                let median_us = median(&latencies);
                times[arm] = median_us;
                let block = Block {
                    pair,
                    arm,
                    order,
                    median_us,
                    elapsed_seconds,
                    correct,
                    distances,
                    fingerprint,
                    load_before,
                    load_after,
                };
                println!(
                    "AB_BLOCK trial={trial} {}",
                    serde_json::to_string(&block).map_err(|e| Error::Backend(e.to_string()))?
                );
                blocks.push(block);
            }
            paired.push(times);
        }
        let a: Vec<_> = paired.iter().map(|p| p[0]).collect();
        let b: Vec<_> = paired.iter().map(|p| p[1]).collect();
        let ratio = median(&a) / median(&b);
        let ci = interval(&paired);
        let same_results = blocks
            .iter()
            .all(|b| b.correct == blocks[0].correct && b.fingerprint == blocks[0].fingerprint);
        let counts_available = blocks[0].distances.is_some();
        let same_counts = blocks.iter().all(|b| b.distances == blocks[0].distances);
        let count_verdict = if counts_available {
            if same_counts {
                "true"
            } else {
                "false"
            }
        } else {
            "unavailable"
        };
        let stable = ci[1] / ci[0] <= 1.10 && same_results && same_counts;
        let after = load()?;
        println!("AB_STABLE={} trial={trial} comparison={} ratio_new_over_b={ratio:.6} ci95=[{:.6},{:.6}] recall={:.6} same_results={same_results} same_counts={count_verdict} load_before={before:?} load_after={after:?}", if stable { "yes" } else { "no" }, if old_b { "new/old" } else { "new/new" }, ci[0], ci[1], blocks[0].correct as f64 / wanted as f64);
        reports.push(serde_json::json!({"trial":trial,"ratio":ratio,"ci95":ci,"stable":stable,"same_results":same_results,"recall_at_10":blocks[0].correct as f64 / wanted as f64,"same_counts":counts_available.then_some(same_counts),"load_before":before,"load_after":after,"blocks":blocks}));
    }
    Ok(
        serde_json::json!({"method":"paired-block-bootstrap-v1","comparison":if old_b {"new/old"} else {"new/new"},"dataset":data.description,"ef":ef,"truth_seconds":truth_seconds,"build_seconds":build_seconds,"counter_compiled":cfg!(feature="native-counter"),"trials":reports}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paired_interval_preserves_exact_ratios() {
        let pairs: Vec<_> = (1..=10).map(|i| [i as f64 * 2.0, i as f64]).collect();
        assert_eq!(interval(&pairs), [2.0, 2.0]);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }
}
