//! Isolate the benchmark's retained input Dataset from the SQLite lookup itself.
use quickdb_api::conformance::Lcg;
use quickdb_api::*;
use quickdb_native::NativeStore;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

fn memory() -> String {
    std::fs::read_to_string("/proc/self/status")
        .map(|s| {
            s.lines()
                .filter(|line| line.starts_with("VmRSS:") || line.starts_with("VmHWM:"))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default()
}

fn lookup(store: &NativeStore, phase: &str, n: u64) -> Result<()> {
    let mut rng = Lcg(0x5142494453);
    let mut times = Vec::with_capacity(10000);
    for _ in 0..10000 {
        let id = u64::from(rng.next_u32()) % n;
        let t = Instant::now();
        let row = store
            .get_row("docs", id)?
            .ok_or_else(|| Error::Backend("missing probe row".into()))?;
        times.push(t.elapsed().as_nanos());
        black_box(row);
    }
    times.sort_unstable();
    println!(
        "POINT_RETAIN phase={phase} median_us={:.3} p95_us={:.3} {}",
        times[5000] as f64 / 1000.0,
        times[9500] as f64 / 1000.0,
        memory()
    );
    Ok(())
}

fn run() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let path = Path::new(args.first().map_or("/data/point-million", String::as_str));
    let n: usize = args.get(1).map_or(Ok(1_000_000), |v| v.parse())?;
    if n == 0 {
        return Err("count must be positive".into());
    }
    let opts = OpenOptions {
        create: false,
        sync_each_write: false,
    };
    let store = NativeStore::open(path, &opts)?;
    lookup(&store, "without_dataset", n as u64)?;
    let start = Instant::now();
    let data = quickdb_bench::synthetic(n, 128)?;
    println!(
        "POINT_RETAIN dataset_rows={} vectors={} edges={} build_s={:.3} {}",
        data.rows.len(),
        data.vectors.len(),
        data.edges.len(),
        start.elapsed().as_secs_f64(),
        memory()
    );
    lookup(&store, "with_dataset", n as u64)?;
    drop(store);
    let store = NativeStore::open(path, &opts)?;
    lookup(&store, "reopen_with_dataset", n as u64)?;
    black_box(&data);
    drop(data);
    lookup(&store, "after_dataset_drop", n as u64)?;
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("POINT_RETAIN ERROR {e}");
        std::process::exit(1);
    }
}
