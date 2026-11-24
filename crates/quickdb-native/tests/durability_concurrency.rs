//! Area 3: concurrent use. In-process: what the type system allows, and a stress test of it.
//! Across processes: one writer with eight readers, and two writers on one directory.
#[path = "durability_common/mod.rs"]
mod common;

use common::*;
use quickdb_api::*;
use quickdb_native::NativeStore;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn data_root() -> PathBuf {
    std::env::var_os("QUICKDB_DATA").map_or_else(std::env::temp_dir, PathBuf::from)
}
fn example(name: &str) -> PathBuf {
    let exe = std::env::current_exe().expect("test binary path");
    exe.parent()
        .and_then(Path::parent)
        .map(|p| p.join("examples").join(name))
        .expect("target layout")
}

// rusqlite::Connection is Send but not Sync, so NativeStore is Send and not Sync.
// The negative half is checked below without a compile failure: a blanket trait with an
// inherent-impl override resolves to `true` only when the bound holds.
fn assert_send<T: Send>() {}
#[test]
fn native_store_is_send() {
    assert_send::<NativeStore>();
}

struct Probe<T>(std::marker::PhantomData<T>);
trait NotSync {
    const SYNC: bool = false;
}
impl<T> NotSync for Probe<T> {}
impl<T: Sync> Probe<T> {
    const SYNC: bool = true;
}
// Compile-time: this file stops compiling if NativeStore ever becomes Sync.
const _: () = assert!(!Probe::<NativeStore>::SYNC && Probe::<std::sync::Mutex<NativeStore>>::SYNC);

fn fill(dir: &Path, cfg: &Config, batches: u64) -> NativeStore {
    let _ = std::fs::remove_dir_all(dir);
    let mut s = open(dir, true).expect("create");
    s.define_table(&table_def()).expect("table");
    s.define_vector_index(&index_def(cfg.dim)).expect("index");
    for n in 0..batches {
        s.upsert_rows(TABLE, &cfg.rows(n)).expect("rows");
        let v = cfg.vectors(n);
        let items: Vec<_> = v.iter().map(|(id, x)| (*id, x.as_slice())).collect();
        s.upsert_vectors(INDEX, &items).expect("vectors");
        s.upsert_edges(&cfg.edges(n)).expect("edges");
    }
    s.flush().expect("flush");
    s
}

/// Not Sync, so threads share it through a Mutex (serialised), or each thread opens its own
/// read-only handle on the same directory (parallel). Both are exercised; answers must match.
#[test]
fn threads_share_through_mutex_and_own_handles() {
    let mut cfg = Config::new(&data_root().join("threads"));
    cfg.seed = 3;
    let s = fill(&cfg.dir.clone(), &cfg, 5);
    let shared = std::sync::Mutex::new(s);
    let total = cfg.batch * 5;
    let t = std::time::Instant::now();
    let (locked, own) = std::thread::scope(|sc| {
        let a: Vec<_> = (0..4)
            .map(|w| {
                let (shared, cfg) = (&shared, &cfg);
                sc.spawn(move || {
                    let mut ok = 0u64;
                    for q in 0..300u64 {
                        let id = (q * 7 + w) % total;
                        let g = shared.lock().expect("no poisoning");
                        if has_vector(&g, cfg, id).expect("knn")
                            && g.get_row(TABLE, id).expect("row").is_some()
                        {
                            ok += 1;
                        }
                    }
                    ok
                })
            })
            .collect();
        let b: Vec<_> = (0..4)
            .map(|w| {
                let cfg = &cfg;
                sc.spawn(move || {
                    let mine = NativeStore::open_read_only(&cfg.dir).expect("second handle");
                    let mut ok = 0u64;
                    for q in 0..300u64 {
                        let id = (q * 11 + w) % total;
                        if has_vector(&mine, cfg, id).expect("knn")
                            && mine.get_row(TABLE, id).expect("row").is_some()
                        {
                            ok += 1;
                        }
                    }
                    ok
                })
            })
            .collect();
        (
            a.into_iter()
                .map(|h| h.join().expect("thread"))
                .sum::<u64>(),
            b.into_iter()
                .map(|h| h.join().expect("thread"))
                .sum::<u64>(),
        )
    });
    println!(
        "THREADS mutex_ok={locked} own_handle_ok={own} of 1200 each, {:.0} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    assert_eq!((locked, own), (1200, 1200));
    drop(shared);
    let _ = std::fs::remove_dir_all(&cfg.dir);
}

/// One writer process, `readers` reader processes, `secs` seconds. Returns all output lines.
fn writer_and_readers(tag: &str, readers: usize, secs: u64, sleep_ms: u64) -> Vec<String> {
    let mut cfg = Config::new(&data_root().join(format!("mp-{tag}")));
    let _ = std::fs::remove_dir_all(&cfg.dir);
    cfg.seed = 17;
    cfg.batch = 100;
    cfg.flush_every = 2;
    cfg.run_secs = secs;
    cfg.sleep_ms = sleep_ms;
    // Seed a few batches so readers have something to open.
    let mut seed_cfg = cfg.clone();
    seed_cfg.max_batches = 2;
    seed_cfg.run_secs = 0;
    let st = Command::new(example("dur_writer"))
        .args(seed_cfg.to_args())
        .output()
        .expect("seed writer");
    assert!(
        st.status.success(),
        "{}",
        String::from_utf8_lossy(&st.stdout)
    );
    cfg.start = 2;
    let writer = Command::new(example("dur_writer"))
        .args(cfg.to_args())
        .env("QDB_TOLERATE_FLUSH", "1")
        .stdout(Stdio::piped())
        .spawn()
        .expect("writer");
    let reader_args = |c: &Config| {
        vec![
            c.dir.display().to_string(),
            c.seed.to_string(),
            c.batch.to_string(),
            c.dim.to_string(),
            secs.to_string(),
            "1500".to_string(),
        ]
    };
    let rs: Vec<_> = (0..readers)
        .map(|_| {
            Command::new(example("dur_reader"))
                .args(reader_args(&cfg))
                .stdout(Stdio::piped())
                .spawn()
                .expect("reader")
        })
        .collect();
    let mut lines = Vec::new();
    let out = writer.wait_with_output().expect("writer output");
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let committed = text.lines().filter(|l| l.starts_with("COMMITTED")).count();
    let flushed = text.lines().filter(|l| l.starts_with("FLUSHED")).count();
    let flush_err: Vec<_> = text
        .lines()
        .filter(|l| l.starts_with("FLUSH_ERROR") || l.starts_with("ERROR"))
        .collect();
    lines.push(format!(
        "WRITER exit={:?} committed_calls={committed} flushed={flushed} flush_errors={} writes_per_s={:.0} first_error={:?}",
        out.status.code(),
        flush_err.len(),
        committed as f64 / secs as f64,
        flush_err.first()
    ));
    for r in rs {
        let o = r.wait_with_output().expect("reader output");
        for l in BufReader::new(o.stdout.as_slice())
            .lines()
            .map_while(std::result::Result::ok)
        {
            lines.push(l);
        }
    }
    // Readers must not have changed what the writer made durable.
    let mut after = cfg.clone();
    after.start = 0;
    let top = text
        .lines()
        .filter_map(|l| {
            l.strip_prefix("COMMITTED ")?
                .split(' ')
                .next()?
                .parse::<u64>()
                .ok()
        })
        .max();
    match verify(&after, top.map_or(2, |t| t + 2), [None; 3]) {
        Ok(v) => lines.push(format!(
            "FINAL rows={} vectors={} edges={} rebuilds_on_reopen={}",
            v.rows, v.vectors, v.edges, v.rebuilds
        )),
        Err(e) => lines.push(format!("FINAL CORRUPT {e}")),
    }
    let _ = std::fs::remove_dir_all(&cfg.dir);
    for l in &lines {
        println!("{l}");
    }
    lines
}

fn assert_clean(lines: &[String]) {
    for l in lines {
        assert!(!l.starts_with("FINAL CORRUPT"), "{l}");
        if l.starts_with("READER") {
            assert!(l.contains(" wrong=0 "), "{l}");
        }
    }
}

/// D3: readers never see "malformed" (or any error) while the writer flushes. 20 rounds.
#[test]
#[ignore = "long (20 x 6 s); run with --ignored"]
fn readers_see_no_errors_while_writer_flushes() {
    let rounds = std::env::var("QDB_D3_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    for round in 0..rounds {
        let lines = writer_and_readers(&format!("d3-{round}"), 2, 6, 20);
        assert_clean(&lines);
        for l in lines.iter().filter(|l| l.starts_with("READER")) {
            assert!(l.contains(" errors=0 "), "round {round}: {l}");
        }
        println!("D3_ROUND {round} clean");
    }
}

#[test]
fn one_writer_two_readers_short() {
    let lines = writer_and_readers("short", 2, 6, 20);
    assert_clean(&lines);
    for l in lines.iter().filter(|l| l.starts_with("READER")) {
        assert!(l.contains(" errors=0 "), "{l}");
    }
}

#[test]
#[ignore = "long: 1 writer + 8 readers for 60 s; run with --ignored"]
fn one_writer_eight_readers_60s() {
    let lines = writer_and_readers("long", 8, 60, 0);
    assert_clean(&lines);
    for l in lines.iter().filter(|l| l.starts_with("READER")) {
        assert!(l.contains(" errors=0 "), "{l}");
    }
}

/// A flush in the writer while readers hold read transactions: the checkpoint cannot truncate.
#[test]
#[ignore = "long (10 s): flush must not fail while 4 reader processes query; run with --ignored"]
fn flush_succeeds_while_another_process_reads() {
    let lines = writer_and_readers("flushblock", 4, 10, 0);
    let w = lines
        .iter()
        .find(|l| l.starts_with("WRITER"))
        .expect("writer line");
    assert!(w.contains(" flush_errors=0 "), "{w}");
}

/// Two writer processes on one directory: the second must be refused at open, at once,
/// and the first writer's store must verify clean. Eight rounds.
#[test]
fn two_writers_one_directory() {
    for round in 0..8u64 {
        let dir = data_root().join(format!("two-writers-{round}"));
        let _ = std::fs::remove_dir_all(&dir);
        let mut a = Config::new(&dir);
        a.seed = 21 + round;
        a.batch = 100;
        a.max_batches = 30;
        a.sleep_ms = 20;
        let mut b = a.clone();
        b.id_base = 1_000_000;
        let spawn = |c: &Config| {
            Command::new(example("dur_writer"))
                .args(c.to_args())
                .stdout(Stdio::piped())
                .spawn()
                .expect("writer")
        };
        let mut pa = spawn(&a);
        let mut out_a = BufReader::new(pa.stdout.take().expect("stdout"));
        let mut lines_a = Vec::new();
        // B starts once A holds the store, while A is still writing.
        loop {
            let mut l = String::new();
            if out_a.read_line(&mut l).expect("read a") == 0 {
                break;
            }
            let first_commit = l.starts_with("COMMITTED");
            lines_a.push(l);
            if first_commit {
                break;
            }
        }
        let t = std::time::Instant::now();
        let ob = spawn(&b).wait_with_output().expect("b");
        let b_ms = t.elapsed().as_secs_f64() * 1e3;
        lines_a.extend(out_a.lines().map_while(std::result::Result::ok));
        let status_a = pa.wait().expect("a");
        let text_b = String::from_utf8_lossy(&ob.stdout).to_string();
        let committed_b = text_b
            .lines()
            .filter(|l| l.starts_with("COMMITTED"))
            .count();
        let done_a = lines_a.iter().any(|l| l.starts_with("DONE"));
        let v = verify(&a, a.max_batches + 1, [Some(a.max_batches - 1); 3]);
        println!(
            "TWO_WRITERS round={round} a_exit={:?} a_done={done_a} b_exit={:?} b_ms={b_ms:.1} b_committed={committed_b} b_said={:?} verify_a={:?}",
            status_a.code(),
            ob.status.code(),
            text_b.lines().next(),
            v.as_ref().map(|v| (v.rows, v.vectors, v.edges, v.rebuilds))
        );
        assert_eq!(status_a.code(), Some(0));
        assert!(done_a);
        assert_eq!(ob.status.code(), Some(1), "second writer was not refused");
        assert_eq!(committed_b, 0);
        assert!(
            text_b.contains("store is locked by another process"),
            "{text_b}"
        );
        let v = v.expect("first writer's store must verify clean");
        assert_eq!((v.rows, v.vectors, v.edges), (30, 30, 30));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// A read-only handle beside a live writer: writes are refused, reads work.
#[test]
fn read_only_handle_refuses_writes() {
    let mut cfg = Config::new(&data_root().join("read-only"));
    cfg.seed = 4;
    let mut w = fill(&cfg.dir.clone(), &cfg, 2);
    let mut r = NativeStore::open_read_only(&cfg.dir).expect("read-only open beside writer");
    assert!(
        matches!(open(&cfg.dir, false), Err(Error::Backend(m)) if m.contains("store is locked by another process"))
    );
    assert_eq!(r.count_rows(TABLE).expect("rows"), 2 * cfg.batch);
    assert!(has_vector(&r, &cfg, 0).expect("knn"));
    assert!(matches!(
        r.upsert_rows(TABLE, &cfg.rows(5)),
        Err(Error::Backend(_))
    ));
    let v = cfg.vectors(5);
    let items: Vec<_> = v.iter().map(|(id, x)| (*id, x.as_slice())).collect();
    assert!(matches!(
        r.upsert_vectors(INDEX, &items),
        Err(Error::Backend(_))
    ));
    assert!(matches!(
        r.upsert_edges(&cfg.edges(5)),
        Err(Error::Backend(_))
    ));
    assert!(matches!(r.delete_rows(TABLE, &[0]), Err(Error::Backend(_))));
    assert!(matches!(
        r.define_table(&table_def()),
        Err(Error::Backend(_))
    ));
    assert!(matches!(
        r.define_vector_index(&index_def(cfg.dim)),
        Err(Error::Backend(_))
    ));
    assert!(matches!(r.flush(), Err(Error::Backend(_))));
    w.upsert_rows(TABLE, &cfg.rows(2))
        .expect("writer still writes");
    w.flush().expect("writer flushes beside a reader");
    assert_eq!(r.count_rows(TABLE).expect("rows"), 3 * cfg.batch);
    drop(w);
    drop(r);
    // The lock dies with the handle: a new writer opens at once.
    open(&cfg.dir, false).expect("reopen after writer dropped");
    let _ = std::fs::remove_dir_all(&cfg.dir);
}

/// Reader open with a stale snapshot must not write the store.
#[test]
fn reader_open_does_not_write_files() {
    let mut cfg = Config::new(&data_root().join("reader-nowrite"));
    cfg.seed = 9;
    let s = fill(&cfg.dir.clone(), &cfg, 2);
    drop(s);
    std::fs::remove_file(cfg.dir.join("emb.hnsw")).expect("rm snapshot");
    let before: Vec<_> = list(&cfg.dir);
    let r = NativeStore::open_read_only(&cfg.dir).expect("open");
    assert_eq!(r.index_rebuilds(), 1);
    let during = list(&cfg.dir);
    drop(r);
    println!("READER_OPEN files before={before:?} during={during:?}");
    assert!(
        !during.iter().any(|(n, _)| n.starts_with("emb.hnsw")),
        "a reader wrote a snapshot"
    );
    let sqlite = |v: &[(String, u64)]| v.iter().find(|(n, _)| n == "quickdb.sqlite").map(|x| x.1);
    assert_eq!(sqlite(&before), sqlite(&during));
    let _ = std::fs::remove_dir_all(&cfg.dir);
}

fn list(dir: &Path) -> Vec<(String, u64)> {
    let mut v: Vec<_> = std::fs::read_dir(dir)
        .expect("dir")
        .map_while(std::result::Result::ok)
        .map(|e| {
            (
                e.file_name().to_string_lossy().to_string(),
                e.metadata().map_or(0, |m| m.len()),
            )
        })
        .collect();
    v.sort();
    v
}

#[test]
fn open_creates_missing_dir_only_with_create() {
    let dir = data_root().join("nested/new/store");
    let _ = std::fs::remove_dir_all(data_root().join("nested"));
    assert!(NativeStore::open(
        &dir,
        &OpenOptions {
            create: false,
            sync_each_write: false
        }
    )
    .is_err());
    assert!(NativeStore::open(&dir, &OpenOptions::default()).is_ok());
    let _ = std::fs::remove_dir_all(data_root().join("nested"));
}
