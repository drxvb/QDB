//! Larger-than-memory probe: loads `<count>` vectors of `<dim>` dimensions in batches and
//! reports progress with the peak resident memory (VmHWM) from /proc.
//! Usage: dur_bulk <dir> <count> <dim> <batch>
use quickdb_api::conformance::Lcg;
use quickdb_api::*;
use quickdb_native::NativeStore;
use std::time::Instant;

fn peak_mb() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .map_or_else(|| "?".into(), |kb| format!("{}", kb / 1024))
}

fn run(args: &[String]) -> Result<()> {
    let num = |i: usize, d: u64| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let dir = std::path::PathBuf::from(args.first().map_or("/data/bulk", |s| s.as_str()));
    let (count, dim, batch) = (num(1, 200_000), num(2, 1024) as usize, num(3, 5000).max(1));
    let t = Instant::now();
    let mut s = NativeStore::open(
        &dir,
        &OpenOptions {
            create: true,
            sync_each_write: false,
        },
    )?;
    s.define_vector_index(&VectorIndexDef {
        name: "emb".into(),
        dim,
        metric: Metric::Cosine,
        m: 16,
        ef_construction: 100,
    })?;
    let mut rng = Lcg(42);
    let mut done = 0u64;
    while done < count {
        let n = batch.min(count - done);
        let data: Vec<Vec<f32>> = (0..n).map(|_| rng.vector(dim)).collect();
        let items: Vec<_> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (done + i as u64, v.as_slice()))
            .collect();
        s.upsert_vectors("emb", &items)?;
        done += n;
        println!(
            "BULK loaded={done} secs={:.1} peak_mb={}",
            t.elapsed().as_secs_f64(),
            peak_mb()
        );
    }
    let f = Instant::now();
    s.flush()?;
    println!(
        "BULK flushed secs={:.1} flush_secs={:.1} peak_mb={}",
        t.elapsed().as_secs_f64(),
        f.elapsed().as_secs_f64(),
        peak_mb()
    );
    drop(s);
    let o = Instant::now();
    let s = NativeStore::open(
        &dir,
        &OpenOptions {
            create: false,
            sync_each_write: false,
        },
    )?;
    println!(
        "BULK reopened count={} rebuilds={} reopen_secs={:.1} peak_mb={} disk_mb={}",
        s.count_vectors("emb")?,
        s.index_rebuilds(),
        o.elapsed().as_secs_f64(),
        peak_mb(),
        dir_size(&dir)? / (1 << 20)
    );
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        println!("BULK ERROR {e} peak_mb={}", peak_mb());
        std::process::exit(1);
    }
    println!("BULK DONE");
}
