//! Layer-0 reachability probe: build once, then count, repair and measure each policy on copies.
//! Run remotely, one process per data set:
//!   cargo run -p quickdb-native --release --example reach_probe -- sift100k|sift1m|corpus
//! Prints one `REACH_PROBE {json}` line per data set and policy. Corpus lines carry row
//! positions and numbers only.
#[cfg(not(test))]
#[allow(dead_code)]
#[path = "../src/hnsw.rs"]
mod hnsw;

#[cfg(not(test))]
mod probe {
    use super::hnsw::{
        reach::{Policy, Repair},
        Hnsw,
    };
    use quickdb_api::{conformance::Lcg, Metric};
    use std::fs::File;
    use std::io::{BufReader, Read};
    use std::time::Instant;

    type Failure = Box<dyn std::error::Error>;
    const EFS: [usize; 4] = [32, 64, 128, 256];
    const QUERIES: usize = 1000;

    fn records(path: &str, n: usize) -> Result<Vec<Vec<u32>>, Failure> {
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

    fn floats(path: &str, n: usize) -> Result<Vec<Vec<f32>>, Failure> {
        Ok(records(path, n)?
            .into_iter()
            .map(|r| r.into_iter().map(f32::from_bits).collect())
            .collect())
    }

    /// Top-level `"dim"` of meta.json; the file is not echoed anywhere.
    fn meta_dim(path: &str) -> Result<usize, Failure> {
        let text = std::fs::read_to_string(path)?;
        let at = text.find("\"dim\"").ok_or("meta.json has no dim")?;
        let rest = text[at + 5..]
            .trim_start()
            .trim_start_matches(':')
            .trim_start();
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        let dim: usize = digits.parse()?;
        if !(1..=4096).contains(&dim) {
            return Err("meta.json dim out of range".into());
        }
        Ok(dim)
    }

    struct Data {
        name: &'static str,
        metric: Metric,
        base: Vec<Vec<f32>>,
        queries: Vec<Vec<f32>>,
        /// Ids are row positions plus one.
        truth: Option<Vec<Vec<u64>>>,
    }

    fn load(which: &str) -> Result<Data, Failure> {
        match which {
            "sift100k" | "sift1m" => {
                let n = if which == "sift1m" {
                    1_000_000
                } else {
                    100_000
                };
                let base = floats("/datasets/sift/sift_base.fvecs", n)?;
                let queries = floats("/datasets/sift/sift_query.fvecs", QUERIES)?;
                let truth = (n == 1_000_000)
                    .then(|| records("/datasets/sift/sift_groundtruth.ivecs", QUERIES))
                    .transpose()?
                    .map(|rows| {
                        rows.into_iter()
                            .map(|r| r.into_iter().take(10).map(|id| u64::from(id) + 1).collect())
                            .collect()
                    });
                Ok(Data {
                    name: if n == 1_000_000 { "sift1m" } else { "sift100k" },
                    metric: Metric::L2,
                    base,
                    queries,
                    truth,
                })
            }
            "corpus" => {
                let dim = meta_dim("/datasets/corpus/meta.json")?;
                let mut bytes = Vec::new();
                File::open("/datasets/corpus/vectors.f32")?.read_to_end(&mut bytes)?;
                if bytes.is_empty() || !bytes.len().is_multiple_of(dim * 4) {
                    return Err("vectors.f32 length is not a multiple of dim".into());
                }
                let base: Vec<Vec<f32>> = bytes
                    .chunks_exact(dim * 4)
                    .map(|row| {
                        row.chunks_exact(4)
                            .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
                            .collect()
                    })
                    .collect();
                if base.iter().flatten().any(|v| !v.is_finite()) {
                    return Err("non-finite corpus vector".into());
                }
                // 1,000 distinct random rows, their stored vectors as queries.
                let mut order: Vec<usize> = (0..base.len()).collect();
                let mut rng = Lcg(0x7eac);
                for i in (1..order.len()).rev() {
                    order.swap(i, rng.next_u32() as usize % (i + 1));
                }
                let queries = order
                    .iter()
                    .take(QUERIES.min(base.len()))
                    .map(|&row| base[row].clone())
                    .collect();
                Ok(Data {
                    name: "corpus",
                    metric: Metric::Cosine,
                    base,
                    queries,
                    truth: None,
                })
            }
            _ => Err("data set must be sift100k, sift1m or corpus".into()),
        }
    }

    fn peak_rss_kb() -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmHWM:"))
                    .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
            })
            .unwrap_or(0)
    }

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    fn quality(h: &Hnsw, data: &Data, truth: &[Vec<u64>]) -> String {
        let mut parts = Vec::new();
        for ef in EFS {
            let mut times = Vec::with_capacity(data.queries.len());
            let mut found = 0usize;
            for (q, wanted) in data.queries.iter().zip(truth) {
                let start = Instant::now();
                let hits = h.search(q, 10, ef);
                times.push(start.elapsed().as_secs_f64() * 1e6);
                found += hits.iter().filter(|hit| wanted.contains(&hit.id)).count();
            }
            times.sort_by(f64::total_cmp);
            let p = |q: f64| times[((times.len() as f64 * q) as usize).min(times.len() - 1)];
            parts.push(format!(
                "\"recall_ef{ef}\":{:.4},\"p50_us_ef{ef}\":{:.1},\"p99_us_ef{ef}\":{:.1}",
                found as f64 / (truth.len() * 10) as f64,
                p(0.50),
                p(0.99)
            ));
        }
        parts.join(",")
    }

    pub fn run() -> Result<(), Failure> {
        let which = std::env::args().nth(1).unwrap_or_else(|| "sift100k".into());
        for var in [
            "QUICKDB_REACH_REPAIR",
            "QUICKDB_REACH_PLANT",
            "QUICKDB_REACH_REPORT",
        ] {
            if std::env::var_os(var).is_some() {
                return Err(
                    format!("{var} must be unset: the probe applies policies itself").into(),
                );
            }
        }
        let start = Instant::now();
        let data = load(&which)?;
        let dim = data.base[0].len();
        println!(
            "REACH_LOAD set={} n={} dim={dim} queries={} seconds={:.3}",
            data.name,
            data.base.len(),
            data.queries.len(),
            start.elapsed().as_secs_f64()
        );
        let mut h = Hnsw::new(dim, data.metric, 16, 200);
        let start = Instant::now();
        for (batch_no, batch) in data.base.chunks(1000).enumerate() {
            let items: Vec<_> = batch
                .iter()
                .enumerate()
                .map(|(i, v)| ((batch_no * 1000 + i + 1) as u64, v.as_slice()))
                .collect();
            h.insert_many(&items);
        }
        let build_ms = start.elapsed().as_secs_f64() * 1e3;
        println!("REACH_BUILD set={} build_ms={build_ms:.1}", data.name);
        let truth = match &data.truth {
            Some(t) => t.clone(),
            None => {
                let start = Instant::now();
                let t: Vec<Vec<u64>> = data
                    .queries
                    .iter()
                    .map(|q| h.exact(q, 10, None).into_iter().map(|hit| hit.id).collect())
                    .collect();
                println!(
                    "REACH_TRUTH set={} exact_queries={} seconds={:.3}",
                    data.name,
                    t.len(),
                    start.elapsed().as_secs_f64()
                );
                t
            }
        };
        let count_times: Vec<f64> = (0..3)
            .map(|_| {
                let start = Instant::now();
                std::hint::black_box(h.reach());
                start.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        let count_ms = median(count_times);
        let context = Context {
            data: &data,
            truth: &truth,
            build_ms,
            count_ms,
        };
        scenario(&context, "natural", &h, &[])?;
        // Planted stress: the natural graph may strand nothing, which leaves repair cost
        // unmeasured. Cut every inbound link of 0.1% of random positions (at least 100), then
        // measure the same three policies on that graph.
        let n = data.base.len();
        let mut order: Vec<usize> = (0..n).collect();
        let mut rng = Lcg(0x57a4d);
        for i in (1..n).rev() {
            order.swap(i, rng.next_u32() as usize % (i + 1));
        }
        let mut picked: Vec<usize> = order[..(n / 1000).max(100).min(n - 1)].to_vec();
        picked.sort_unstable();
        let mut cut = h;
        let stranded = cut.strand(&picked);
        println!("REACH_STRAND set={} cut={}", data.name, stranded.len());
        scenario(&context, "stranded", &cut, &stranded)?;
        Ok(())
    }

    struct Context<'a> {
        data: &'a Data,
        truth: &'a [Vec<u64>],
        build_ms: f64,
        count_ms: f64,
    }

    /// For these positions, searched with their own stored vector at ef 64: the share returned
    /// at rank 1, the share returned among the top-10 hits tied with rank 1's distance (an
    /// identical vector with a lower id wins rank 1 by the id tie rule), and how many such
    /// positions had a tie at rank 1.
    fn self_found(h: &Hnsw, data: &Data, positions: &[usize]) -> String {
        if positions.is_empty() {
            return "\"stranded_self_rank1_ef64\":null,\"stranded_self_tied_ef64\":null,\"stranded_rank1_ties\":0".into();
        }
        let (mut rank1, mut tied, mut ties) = (0usize, 0usize, 0usize);
        for &p in positions {
            let id = p as u64 + 1;
            let hits = h.search(&data.base[p], 10, 64);
            let Some(first) = hits.first() else { continue };
            let level: Vec<_> = hits
                .iter()
                .filter(|hit| hit.distance == first.distance)
                .collect();
            rank1 += usize::from(first.id == id);
            tied += usize::from(level.iter().any(|hit| hit.id == id));
            ties += usize::from(level.len() > 1);
        }
        let share = |c: usize| c as f64 / positions.len() as f64;
        format!(
            "\"stranded_self_rank1_ef64\":{:.4},\"stranded_self_tied_ef64\":{:.4},\"stranded_rank1_ties\":{ties}",
            share(rank1),
            share(tied)
        )
    }

    fn scenario(
        context: &Context<'_>,
        label: &str,
        h: &Hnsw,
        stranded: &[usize],
    ) -> Result<(), Failure> {
        let Context {
            data,
            truth,
            build_ms,
            count_ms,
        } = *context;
        let dim = data.base[0].len();
        let base = h.reach();
        for policy in [None, Some(Policy::Reference), Some(Policy::Selective)] {
            let (graph, line) = match policy {
                None => (
                    None,
                    format!(
                        "\"unreachable_before\":{u},\"unreachable_after\":{u},\"zero_inbound_before\":{z},\"zero_inbound_after\":{z},\"bridges\":0,\"global_searches\":0,\"local\":0,\"fallback_any\":0,\"failed\":0,\"repair_ms\":0,\"repair_ms_runs\":[]",
                        u = base.unreachable,
                        z = base.zero_inbound
                    ),
                ),
                Some(policy) => {
                    // Three fresh copies; only the last is kept, so at most two graphs live.
                    let mut runs = Vec::new();
                    let mut kept = None;
                    let mut first: Option<Repair> = None;
                    for run in 0..3 {
                        let mut copy = h.clone();
                        let start = Instant::now();
                        let repair = copy.repair_reach(policy);
                        runs.push(start.elapsed().as_secs_f64() * 1e3);
                        if first.is_some_and(|f| f != repair) {
                            return Err("repair is not deterministic across copies".into());
                        }
                        first = Some(repair);
                        if run == 2 {
                            kept = Some((copy, repair));
                        }
                    }
                    let (copy, r) = kept.ok_or("no repair ran")?;
                    let repair_ms = median(runs.clone());
                    let runs: Vec<String> = runs.iter().map(|ms| format!("{ms:.2}")).collect();
                    (
                        Some(copy),
                        format!(
                            "\"unreachable_before\":{},\"unreachable_after\":{},\"zero_inbound_before\":{},\"zero_inbound_after\":{},\"zero_inbound_bridged_after\":{},\"bridges\":{},\"global_searches\":{},\"local\":{},\"fallback_any\":{},\"failed\":{},\"repair_ms\":{repair_ms:.2},\"repair_ms_runs\":[{}],\"repair_pct_of_build\":{:.4}",
                            r.before.unreachable,
                            r.after.unreachable,
                            r.before.zero_inbound,
                            r.after.zero_inbound,
                            r.after.zero_inbound_bridged,
                            r.bridges,
                            r.global_searches,
                            r.local,
                            r.fallback_any,
                            r.failed,
                            runs.join(","),
                            100.0 * repair_ms / build_ms,
                        ),
                    )
                }
            };
            let graph = graph.as_ref().unwrap_or(h);
            let measured = quality(graph, data, truth);
            println!(
                "REACH_PROBE {{\"set\":\"{}\",\"scenario\":\"{label}\",\"policy\":\"{}\",\"n\":{},\"dim\":{dim},\"metric\":\"{}\",\"m\":16,\"ef_construction\":200,\"build_ms\":{build_ms:.1},\"count_ms\":{count_ms:.2},\"stranded\":{},{line},{},{measured},\"peak_rss_kb\":{}}}",
                data.name,
                Policy::name(policy),
                data.base.len(),
                if data.metric == Metric::L2 { "l2" } else { "cosine" },
                stranded.len(),
                self_found(graph, data, stranded),
                peak_rss_kb()
            );
        }
        Ok(())
    }
}

fn main() {
    #[cfg(not(test))]
    if let Err(e) = probe::run() {
        eprintln!("REACH_PROBE_ERROR {e}");
        std::process::exit(1);
    }
}
