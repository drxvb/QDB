//! Remote-only before/after probe. Build once, then run each call in a fresh process.
use quickdb_api::*;
use quickdb_native::NativeStore;
use std::path::Path;
use std::time::Instant;

const N: usize = 1_000_000;
const STAR: u64 = 2_000_000;

fn open(name: &str) -> NativeStore {
    NativeStore::open(
        Path::new(&format!("/data/bound_{name}")),
        &OpenOptions::default(),
    )
    .expect("open probe store")
}

fn sampled(id: usize) -> bool {
    (0..256).any(|i| i * N / 256 == id)
}

fn allowed(id: usize) -> bool {
    id % 10 < 3 && !sampled(id)
}

fn memory() -> (u64, u64) {
    let status = std::fs::read_to_string("/proc/self/status").expect("Linux process status");
    let field = |name: &str| {
        status
            .lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse().ok())
            .expect("memory field")
    };
    (field("VmHWM:"), field("VmRSS:"))
}

fn measure<T>(name: &str, call: impl FnOnce() -> T) -> T {
    // Linux clear_refs=5 resets the high watermark to current RSS, excluding setup.
    std::fs::write("/proc/self/clear_refs", "5").expect("reset VmHWM");
    let before = memory();
    let start = Instant::now();
    let result = call();
    let elapsed = start.elapsed();
    let after = memory();
    println!("BOUND {name} before_hwm_kib={} before_rss_kib={} after_hwm_kib={} after_rss_kib={} extra_hwm_kib={} elapsed_us={}", before.0, before.1, after.0, after.1, after.0.saturating_sub(before.0), elapsed.as_micros());
    result
}

fn build() {
    let mut s = open("star");
    for base in (1..=STAR).step_by(10_000) {
        let edges: Vec<_> = (base..=(base + 9_999).min(STAR))
            .map(|dst| Edge {
                src: 0,
                dst,
                rel: "r".into(),
                weight: 1.0,
            })
            .collect();
        s.upsert_edges(&edges).expect("star edges");
    }
    let chain: Vec<_> = (STAR..STAR + 100)
        .map(|src| Edge {
            src,
            dst: src + 1,
            rel: "r".into(),
            weight: 1.0,
        })
        .collect();
    s.upsert_edges(&chain).expect("chain behind star");
    s.flush().expect("star flush");
    drop(s);
    println!("BOUND built star edges={}", STAR + 100);
    let mut s = open("filtered");
    s.define_table(&TableDef {
        name: "docs".into(),
        columns: vec![ColumnDef {
            name: "p".into(),
            ty: ColumnType::Int,
            indexed: true,
        }],
    })
    .expect("table");
    s.define_vector_index(&VectorIndexDef {
        name: "emb".into(),
        dim: 16,
        metric: Metric::L2,
        m: 4,
        ef_construction: 8,
    })
    .expect("index");
    for base in (0..N).step_by(10_000) {
        let rows: Vec<_> = (base..(base + 10_000).min(N))
            .map(|id| Row {
                id: id as u64,
                values: vec![Value::Int(i64::from(allowed(id)))],
            })
            .collect();
        s.upsert_rows("docs", &rows).expect("rows");
        let vectors: Vec<_> = (base..(base + 10_000).min(N))
            .map(|id| {
                let mut rng = quickdb_api::conformance::Lcg(id as u64 + 91);
                (id as u64, rng.vector(16))
            })
            .collect();
        let items: Vec<_> = vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        s.upsert_vectors("emb", &items).expect("vectors");
        if base % 100_000 == 0 {
            println!("BOUND vector_build={}", base + 10_000);
        }
    }
    s.flush().expect("vector snapshot");
    println!(
        "BOUND built vectors={N} matching={}",
        (0..N).filter(|&id| allowed(id)).count()
    );
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("build") => build(),
        Some("star") => {
            let s = open("star");
            if std::env::var_os("BOUND_WARM_CACHE").is_some() {
                assert_eq!(s.count_edges().expect("warm edge cache"), STAR + 100);
                println!("BOUND cache=warm edges={}", STAR + 100);
            }
            let hits = measure("star", || {
                s.k_hop(0, None, Direction::Out, 4, 1).expect("BFS")
            });
            assert_eq!(hits, vec![Reached { id: 1, depth: 1 }]);
            println!("BOUND star result={hits:?}");
        }
        Some("filtered") => {
            let s = open("filtered");
            assert_eq!(s.index_rebuilds(), 0);
            if std::env::var_os("BOUND_WARM_CACHE").is_some() {
                assert_eq!(s.count_rows("docs").expect("warm row cache"), N as u64);
                std::hint::black_box(
                    s.knn("emb", &[0.1; 16], 1, 1024)
                        .expect("warm reusable graph scratch"),
                );
                println!("BOUND cache=warm rows={N} graph_scratch=warm");
            }
            for i in 0..256 {
                let row = s
                    .get_row("docs", (i * N / 256) as u64)
                    .expect("sample")
                    .expect("row");
                assert_eq!(row.values[0], Value::Int(0));
            }
            let exact_up_to = (2.0 * (64.0 * N as f64).sqrt()).max(1024.0) as usize;
            println!("BOUND path=masked sample_hits=0 samples=256 lazy_from={} exact_up_to={exact_up_to} mask_bytes={}", (15.0 * 64.0 / N as f64).sqrt().max(0.02), N.div_ceil(64) * 8);
            let hits = measure("filtered", || {
                s.knn_filtered(
                    "emb",
                    &[0.1; 16],
                    1,
                    64,
                    "docs",
                    &Filter::Eq {
                        column: "p".into(),
                        value: Value::Int(1),
                    },
                )
                .expect("filtered KNN")
            });
            assert_eq!(hits.len(), 1);
            println!("BOUND filtered result={hits:?}");
        }
        Some("small") => {
            let mut s = open("small");
            let edges: Vec<_> = (1..=6)
                .map(|dst| Edge {
                    src: 0,
                    dst,
                    rel: "r".into(),
                    weight: 1.0,
                })
                .collect();
            s.upsert_edges(&edges).expect("small edges");
            for _ in 0..1000 {
                std::hint::black_box(s.k_hop(0, None, Direction::Out, 2, 100).expect("warmup"));
            }
            for round in 0..9 {
                let start = Instant::now();
                for _ in 0..50_000 {
                    std::hint::black_box(
                        s.k_hop(0, None, Direction::Out, 2, 100).expect("small BFS"),
                    );
                }
                println!(
                    "BOUND small round={round} ns_per_call={}",
                    start.elapsed().as_nanos() / 50_000
                );
            }
        }
        _ => panic!("use build, star, filtered, or small"),
    }
}
