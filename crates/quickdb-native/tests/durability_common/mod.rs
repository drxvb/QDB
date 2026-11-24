//! Shared by the durability tests and the helper programs in `examples/`:
//! the deterministic content of every batch, the writer configuration and the checker.
// Each test target and example uses a different subset of these helpers.
#![allow(dead_code)]

use quickdb_api::conformance::Lcg;
use quickdb_api::*;
use quickdb_native::NativeStore;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const TABLE: &str = "docs";
pub const INDEX: &str = "emb";
pub const REL: &str = "next";
/// `ef_search` at least the node count makes the native index score every vector exactly.
pub const EXACT_EF: usize = 1 << 30;

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub dir: PathBuf,
    pub sync: bool,
    pub seed: u64,
    pub batch: u64,
    pub dim: usize,
    pub flush_every: u64,
    /// Re-upsert the vectors of batch n-1 (same content) in batch n, so compaction runs.
    pub replace: bool,
    pub start: u64,
    pub max_batches: u64,
    pub id_base: u64,
    pub sleep_ms: u64,
    /// Stop after this many seconds (0 = only `max_batches`).
    pub run_secs: u64,
}

impl Config {
    pub fn new(dir: &Path) -> Self {
        Config {
            dir: dir.to_path_buf(),
            sync: false,
            seed: 1,
            batch: 200,
            dim: 32,
            flush_every: 3,
            replace: false,
            start: 0,
            max_batches: 1_000_000,
            id_base: 0,
            sleep_ms: 0,
            run_secs: 0,
        }
    }
    pub fn to_args(&self) -> Vec<String> {
        vec![
            self.dir.display().to_string(),
            u8::from(self.sync).to_string(),
            self.seed.to_string(),
            self.batch.to_string(),
            self.dim.to_string(),
            self.flush_every.to_string(),
            u8::from(self.replace).to_string(),
            self.start.to_string(),
            self.max_batches.to_string(),
            self.id_base.to_string(),
            self.sleep_ms.to_string(),
            self.run_secs.to_string(),
        ]
    }
    pub fn from_args(args: &[String]) -> std::result::Result<Self, String> {
        if args.len() != 12 {
            return Err(format!("expected 12 arguments, got {}", args.len()));
        }
        let n = |i: usize| -> std::result::Result<u64, String> {
            args[i]
                .parse::<u64>()
                .map_err(|e| format!("argument {i}: {e}"))
        };
        Ok(Config {
            dir: PathBuf::from(&args[0]),
            sync: n(1)? != 0,
            seed: n(2)?,
            batch: n(3)?.max(2),
            dim: n(4)?.max(1) as usize,
            flush_every: n(5)?.max(1),
            replace: n(6)? != 0,
            start: n(7)?,
            max_batches: n(8)?,
            id_base: n(9)?,
            sleep_ms: n(10)?,
            run_secs: n(11)?,
        })
    }
    pub fn id(&self, n: u64, i: u64) -> NodeId {
        self.id_base + n * self.batch + i
    }
    pub fn row(&self, n: u64, i: u64) -> Row {
        Row {
            id: self.id(n, i),
            values: vec![Value::Int(n as i64), Value::Text(format!("b{n}-{i}"))],
        }
    }
    pub fn vector(&self, id: NodeId) -> Vec<f32> {
        Lcg(self.seed ^ id.wrapping_mul(0x9E37_79B9_7F4A_7C15)).vector(self.dim)
    }
    pub fn edge(&self, n: u64, i: u64) -> Edge {
        Edge {
            src: self.id(n, i),
            rel: REL.into(),
            dst: self.id(n, i) + 1,
            weight: n as f32,
        }
    }
    pub fn rows(&self, n: u64) -> Vec<Row> {
        (0..self.batch).map(|i| self.row(n, i)).collect()
    }
    pub fn vectors(&self, n: u64) -> Vec<(NodeId, Vec<f32>)> {
        (0..self.batch)
            .map(|i| {
                let id = self.id(n, i);
                (id, self.vector(id))
            })
            .collect()
    }
    pub fn edges(&self, n: u64) -> Vec<Edge> {
        (0..self.batch).map(|i| self.edge(n, i)).collect()
    }
}

pub fn table_def() -> TableDef {
    TableDef {
        name: TABLE.into(),
        columns: vec![
            ColumnDef {
                name: "batch".into(),
                ty: ColumnType::Int,
                indexed: true,
            },
            ColumnDef {
                name: "label".into(),
                ty: ColumnType::Text,
                indexed: false,
            },
        ],
    }
}

pub fn index_def(dim: usize) -> VectorIndexDef {
    VectorIndexDef {
        name: INDEX.into(),
        dim,
        metric: Metric::L2,
        m: 16,
        ef_construction: 100,
    }
}

pub fn open(dir: &Path, create: bool) -> Result<NativeStore> {
    NativeStore::open(
        dir,
        &OpenOptions {
            create,
            sync_each_write: false,
        },
    )
}

/// True when the exact search finds `id` with its own vector at distance zero.
pub fn has_vector(s: &NativeStore, cfg: &Config, id: NodeId) -> Result<bool> {
    let hits = s.knn(INDEX, &cfg.vector(id), 1, EXACT_EF)?;
    Ok(hits
        .first()
        .is_some_and(|h| h.id == id && h.distance == 0.0))
}

#[derive(Debug, Clone, PartialEq)]
pub struct Verified {
    /// Number of leading batches present, per kind.
    pub rows: u64,
    pub vectors: u64,
    pub edges: u64,
    pub rebuilds: usize,
    pub reopen_ms: f64,
    /// Self-queries through the approximate path that did not return the query's own id.
    pub ann_self_miss: u64,
    pub digest: Vec<String>,
}

/// Checks a store written by the writer helper. `upto` is one past the highest batch number
/// that may have been touched; `must[k]` the highest batch whose kind-k write must be durable
/// (rows, vectors, edges).
pub fn verify(
    cfg: &Config,
    upto: u64,
    must: [Option<u64>; 3],
) -> std::result::Result<Verified, String> {
    let t = Instant::now();
    let s = open(&cfg.dir, false).map_err(|e| format!("open after kill failed: {e}"))?;
    let reopen_ms = t.elapsed().as_secs_f64() * 1e3;
    let rebuilds = s.index_rebuilds();
    let e = |x: Error| x.to_string();
    let mut present = [Vec::new(), Vec::new(), Vec::new()];
    for n in 0..upto {
        let ends = [0, cfg.batch - 1];
        let mut kinds = [[false; 2]; 3];
        for (j, &i) in ends.iter().enumerate() {
            let id = cfg.id(n, i);
            kinds[0][j] = match s.get_row(TABLE, id).map_err(e)? {
                Some(r) if r == cfg.row(n, i) => true,
                Some(r) => return Err(format!("batch {n}: row {id} has wrong content {r:?}")),
                None => false,
            };
            kinds[1][j] = has_vector(&s, cfg, id).map_err(e)?;
            let got = s.neighbors(id, Some(REL), Direction::Out, 10).map_err(e)?;
            kinds[2][j] = match got.as_slice() {
                [] => false,
                [x] if *x == cfg.edge(n, i) => true,
                other => return Err(format!("batch {n}: edges of {id} wrong: {other:?}")),
            };
        }
        for (k, name) in ["rows", "vectors", "edges"].iter().enumerate() {
            if kinds[k][0] != kinds[k][1] {
                return Err(format!(
                    "batch {n}: {name} partly applied (first={}, last={})",
                    kinds[k][0], kinds[k][1]
                ));
            }
            present[k].push(kinds[k][0]);
        }
    }
    let mut prefix = [0u64; 3];
    for (p, kind) in prefix.iter_mut().zip(&present) {
        *p = kind.iter().take_while(|x| **x).count() as u64;
        if kind.iter().skip(*p as usize).any(|x| *x) {
            return Err(format!("present batches are not a prefix: {kind:?}"));
        }
    }
    let [r, v, ed] = prefix;
    if !(r >= v && v >= ed && r - ed <= 1) {
        return Err(format!(
            "write order broken: rows {r}, vectors {v}, edges {ed} batches"
        ));
    }
    for (k, m) in must.iter().enumerate() {
        if let Some(f) = m {
            if prefix[k] < f + 1 {
                return Err(format!(
                    "kind {k}: batch {f} was durable but only {} complete batches survived",
                    prefix[k]
                ));
            }
        }
    }
    let counts = (
        s.count_rows(TABLE).map_err(e)?,
        s.count_vectors(INDEX).map_err(e)?,
        s.count_edges().map_err(e)?,
    );
    if counts != (r * cfg.batch, v * cfg.batch, ed * cfg.batch) {
        return Err(format!(
            "counts {counts:?} do not match batches rows {r} vectors {v} edges {ed}"
        ));
    }
    let mut ann_self_miss = 0;
    let mut digest = vec![format!("{counts:?}")];
    let mut rng = Lcg(cfg.seed ^ upto);
    for _ in 0..8.min(v) {
        let n = rng.next_u32() as u64 % v;
        let id = cfg.id(n, rng.next_u32() as u64 % cfg.batch);
        let hits = s.knn(INDEX, &cfg.vector(id), 10, 200).map_err(e)?;
        if !hits.iter().any(|h| h.id == id) {
            ann_self_miss += 1;
        }
        let exact = s.knn(INDEX, &cfg.vector(id), 10, EXACT_EF).map_err(e)?;
        digest.push(format!(
            "{:?}",
            exact.iter().map(|h| h.id).collect::<Vec<_>>()
        ));
        let filter = Filter::Eq {
            column: "batch".into(),
            value: Value::Int(n as i64),
        };
        let mut got: Vec<_> = s
            .knn_filtered(
                INDEX,
                &cfg.vector(id),
                cfg.batch as usize,
                EXACT_EF,
                TABLE,
                &filter,
            )
            .map_err(e)?
            .into_iter()
            .map(|h| h.id)
            .collect();
        got.sort_unstable();
        let want: Vec<_> = (0..cfg.batch).map(|i| cfg.id(n, i)).collect();
        if got != want {
            return Err(format!(
                "knn_filtered on batch {n} disagrees with the rows ({} of {} ids)",
                got.len(),
                want.len()
            ));
        }
    }
    Ok(Verified {
        rows: r,
        vectors: v,
        edges: ed,
        rebuilds,
        reopen_ms,
        ann_self_miss,
        digest,
    })
}
