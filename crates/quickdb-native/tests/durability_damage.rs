//! Area 2: damaged snapshot and WAL files. Open must succeed with correct answers or fail with an
//! `Error`; it must never panic or answer wrongly.
#[path = "durability_common/mod.rs"]
mod common;

use common::*;
use quickdb_api::conformance::Lcg;
use quickdb_api::*;
use std::fs;
use std::path::{Path, PathBuf};

fn data_root() -> PathBuf {
    std::env::var_os("QUICKDB_DATA").map_or_else(std::env::temp_dir, PathBuf::from)
}

/// A flushed store of `batches` complete batches, written in-process.
fn build(dir: &Path, cfg: &Config, batches: u64) {
    let _ = fs::remove_dir_all(dir);
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
}

fn copy_dir(from: &Path, to: &Path) {
    let _ = fs::remove_dir_all(to);
    fs::create_dir_all(to).expect("mkdir");
    for e in fs::read_dir(from)
        .expect("read_dir")
        .map_while(std::result::Result::ok)
    {
        fs::copy(e.path(), to.join(e.file_name())).expect("copy");
    }
}

#[derive(Default, Debug)]
struct Tally {
    correct: u64,
    rebuilt: u64,
    clean_error: u64,
    wrong: Vec<String>,
    panics: u64,
}

fn check(cfg: &Config, batches: u64, what: &str, t: &mut Tally) {
    let r = std::panic::catch_unwind(|| verify(cfg, batches + 1, [Some(batches - 1); 3]));
    match r {
        Err(_) => {
            t.panics += 1;
            t.wrong.push(format!("{what}: panic"));
        }
        Ok(Ok(v)) => {
            t.correct += 1;
            t.rebuilt += v.rebuilds as u64;
        }
        // An open error is acceptable; any other verification failure is a wrong answer.
        Ok(Err(e)) if e.starts_with("open after kill failed") => t.clean_error += 1,
        Ok(Err(e)) => t.wrong.push(format!("{what}: {e}")),
    }
}

fn damage_campaign(seed: u64, count: u64) -> Tally {
    println!("durability_damage seed={seed} damages={count}");
    let mut cfg = Config::new(&data_root().join(format!("damage-{seed}-work")));
    cfg.seed = seed;
    cfg.batch = 300;
    let batches = 4;
    let pristine = data_root().join(format!("damage-{seed}-pristine"));
    build(&pristine, &cfg, batches);
    let snap_len = fs::metadata(pristine.join("emb.hnsw"))
        .expect("snapshot")
        .len();
    let mut rng = Lcg(seed);
    let mut t = Tally::default();
    for k in 0..count {
        copy_dir(&pristine, &cfg.dir);
        let snap = cfg.dir.join("emb.hnsw");
        let what = match k % 4 {
            0 => {
                let len = rng.next_u32() as u64 % snap_len;
                fs::OpenOptions::new()
                    .write(true)
                    .open(&snap)
                    .and_then(|f| f.set_len(len))
                    .expect("truncate");
                format!("truncate to {len}")
            }
            1 | 2 => {
                let mut b = fs::read(&snap).expect("read");
                let flips = 1 + rng.next_u32() % 8;
                for _ in 0..flips {
                    let at = rng.next_u32() as usize % b.len();
                    b[at] ^= 1 << (rng.next_u32() % 8);
                }
                fs::write(&snap, b).expect("write");
                format!("flip {flips} bits")
            }
            _ => {
                fs::remove_file(&snap).expect("rm");
                "delete snapshot".into()
            }
        };
        check(&cfg, batches, &what, &mut t);
    }
    // After a clean flush the WAL is checkpointed; removing it (and the shm) must lose nothing.
    for name in ["quickdb.sqlite-wal", "quickdb.sqlite-shm"] {
        copy_dir(&pristine, &cfg.dir);
        let _ = fs::remove_file(cfg.dir.join(name));
        check(&cfg, batches, &format!("delete {name}"), &mut t);
    }
    println!(
        "DAMAGE correct={} of them rebuilt={} clean_error={} wrong={} panics={}",
        t.correct,
        t.rebuilt,
        t.clean_error,
        t.wrong.len(),
        t.panics
    );
    for w in t.wrong.iter().take(5) {
        println!("  WRONG {w}");
    }
    let _ = fs::remove_dir_all(&cfg.dir);
    let _ = fs::remove_dir_all(&pristine);
    t
}

#[test]
fn damage_short() {
    let t = damage_campaign(5, 24);
    assert!(t.wrong.is_empty(), "{:?}", t.wrong);
}

#[test]
#[ignore = "long: 200 random damages; run with --ignored"]
fn damage_long() {
    let seed = std::env::var("QDB_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(99);
    let t = damage_campaign(seed, 200);
    assert!(t.wrong.is_empty(), "{:?}", t.wrong);
}

/// A same-revision snapshot with a valid checksum but from another store must be rejected
/// by its store id and rebuilt from SQLite.
#[test]
fn foreign_snapshot_with_same_revision() {
    for round in 0..8u64 {
        let mut a = Config::new(&data_root().join("foreign-a"));
        a.seed = 1 + 2 * round;
        let mut b = Config::new(&data_root().join("foreign-b"));
        b.seed = 2 + 2 * round;
        build(&a.dir, &a, 2);
        build(&b.dir, &b, 2);
        fs::copy(b.dir.join("emb.hnsw"), a.dir.join("emb.hnsw")).expect("copy");
        let r = verify(&a, 3, [Some(1); 3]);
        println!(
            "FOREIGN round={round} snapshot rebuilds: {:?}",
            r.as_ref().map(|v| v.rebuilds)
        );
        let _ = fs::remove_dir_all(&a.dir);
        let _ = fs::remove_dir_all(&b.dir);
        let v = r.expect("foreign snapshot gave wrong answers");
        assert_eq!(v.rebuilds, 1, "foreign snapshot was trusted");
    }
}

fn copy_sqlite(from: &Path, to: &Path) {
    for name in ["quickdb.sqlite", "quickdb.sqlite-wal", "quickdb.sqlite-shm"] {
        let _ = fs::remove_file(to.join(name));
        if from.join(name).exists() {
            fs::copy(from.join(name), to.join(name)).expect("copy sqlite");
        }
    }
}

fn write_batches(cfg: &Config, range: std::ops::Range<u64>) {
    let mut s = open(&cfg.dir, false).expect("open");
    for n in range {
        s.upsert_rows(TABLE, &cfg.rows(n)).expect("rows");
        let v = cfg.vectors(n);
        let items: Vec<_> = v.iter().map(|(id, x)| (*id, x.as_slice())).collect();
        s.upsert_vectors(INDEX, &items).expect("vectors");
        s.upsert_edges(&cfg.edges(n)).expect("edges");
    }
    s.flush().expect("flush");
}

/// Same store, same revision, different data: a quickdb.sqlite restored from backup beside a
/// newer snapshot whose revision counter happens to match. The nonce must reject it.
#[test]
fn restored_backup_beside_newer_snapshot() {
    for round in 0..8u64 {
        let mut cfg = Config::new(&data_root().join("restore-live"));
        cfg.seed = 30 + round;
        let mut other = cfg.clone();
        other.seed = 1000 + round;
        let b0 = data_root().join("restore-b0");
        let b1 = data_root().join("restore-b1");
        for d in [&b0, &b1] {
            let _ = fs::remove_dir_all(d);
            fs::create_dir_all(d).expect("mkdir");
        }
        build(&cfg.dir, &cfg, 2);
        copy_sqlite(&cfg.dir, &b0);
        write_batches(&cfg, 2..4);
        copy_sqlite(&cfg.dir, &b1);
        // Back to backup 0, then a different history to the same revision count.
        copy_sqlite(&b0, &cfg.dir);
        write_batches(&other, 2..4);
        // Now restore backup 1 beside the snapshot of the other history.
        copy_sqlite(&b1, &cfg.dir);
        let r = verify(&cfg, 5, [Some(3); 3]);
        println!(
            "RESTORED_BACKUP round={round} rebuilds: {:?}",
            r.as_ref().map(|v| v.rebuilds)
        );
        for d in [&cfg.dir, &b0, &b1] {
            let _ = fs::remove_dir_all(d);
        }
        let v = r.expect("restored backup gave wrong answers");
        assert_eq!(v.rebuilds, 1, "snapshot of another history was trusted");
    }
}

#[test]
fn reopen_of_missing_store_without_create_is_error() {
    let dir = data_root().join("missing-store-x");
    let _ = fs::remove_dir_all(&dir);
    assert!(matches!(open(&dir, false), Err(Error::Io(_))));
}
