//! Structured API fuzzing: arbitrary sequences of schema, row, vector and edge operations,
//! searches, deletes, flushes and reopens on a temporary store. No operation may panic, and:
//! - `knn` and `knn_filtered` return at most `k` hits, sorted by distance, with distinct ids;
//! - `k_hop` returns at most `limit` results, distinct, never the start node, within depth;
//! - `select` and `neighbors` return at most `limit` results.
#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use quickdb_api::*;
use quickdb_native::NativeStore;

// Small name pools so operations meet each other; the last entries are invalid or reserved.
const TABLES: [&str; 3] = ["ta", "tb", "Bad"];
const COLUMNS: [&str; 5] = ["x", "y", "z", "id", "9"];
const INDEXES: [&str; 3] = ["va", "vb", "no such"];
const RELS: [&str; 3] = ["r", "s", ""];

fn pick<'a>(pool: &[&'a str], i: u8) -> &'a str {
    pool[i as usize % pool.len()]
}

#[derive(Arbitrary, Debug)]
enum Id {
    Small(u8),
    Raw(u64),
}
impl Id {
    fn get(&self) -> NodeId {
        match *self {
            Id::Small(v) => u64::from(v % 48),
            Id::Raw(v) => v,
        }
    }
}

#[derive(Arbitrary, Debug)]
enum Val {
    Null,
    Int(i64),
    SmallInt(i8),
    Float(f64),
    Text(String),
    Bytes(Vec<u8>),
}
impl Val {
    fn get(&self) -> Value {
        match self {
            Val::Null => Value::Null,
            Val::Int(v) => Value::Int(*v),
            Val::SmallInt(v) => Value::Int(i64::from(*v)),
            Val::Float(v) => Value::Float(*v),
            Val::Text(v) => Value::Text(v.clone()),
            Val::Bytes(v) => Value::Bytes(v.clone()),
        }
    }
}

#[derive(Arbitrary, Debug)]
enum Filt {
    All,
    Eq(u8, Val),
    IntRange(u8, i64, i64),
    SmallRange(u8, i8, i8),
    TextPrefix(u8, String),
}
impl Filt {
    fn get(&self) -> Filter {
        let column = |c: &u8| pick(&COLUMNS, *c).to_string();
        match self {
            Filt::All => Filter::All,
            Filt::Eq(c, v) => Filter::Eq {
                column: column(c),
                value: v.get(),
            },
            Filt::IntRange(c, min, max) => Filter::IntRange {
                column: column(c),
                min: *min,
                max: *max,
            },
            Filt::SmallRange(c, min, max) => Filter::IntRange {
                column: column(c),
                min: i64::from(*min),
                max: i64::from(*max),
            },
            Filt::TextPrefix(c, p) => Filter::TextPrefix {
                column: column(c),
                prefix: p.clone(),
            },
        }
    }
}

#[derive(Arbitrary, Debug)]
enum Coord {
    Small(i8),
    Raw(f32),
    Nan,
    Inf,
}
impl Coord {
    fn get(&self) -> f32 {
        match *self {
            Coord::Small(v) => f32::from(v) * 0.5,
            Coord::Raw(v) => v,
            Coord::Nan => f32::NAN,
            Coord::Inf => f32::INFINITY,
        }
    }
}

/// A vector with the index's dimension most of the time, and occasionally a wrong one.
#[derive(Arbitrary, Debug)]
struct Vector {
    wrong_dim: Option<u8>,
    coords: Vec<Coord>,
}
impl Vector {
    fn get(&self, dim: usize) -> Vec<f32> {
        let len = self.wrong_dim.map_or(dim, |d| usize::from(d % 9));
        (0..len)
            .map(|i| self.coords.get(i).map_or(i as f32, Coord::get))
            .collect()
    }
}

#[derive(Arbitrary, Debug)]
enum Size {
    Small(u8),
    Raw(usize),
}
impl Size {
    fn get(&self) -> usize {
        match *self {
            Size::Small(v) => usize::from(v),
            Size::Raw(v) => v,
        }
    }
}

#[derive(Arbitrary, Debug)]
enum Op {
    DefineTable {
        table: u8,
        columns: Vec<(u8, u8, bool)>,
    },
    UpsertRows {
        table: u8,
        rows: Vec<(Id, Vec<Val>)>,
    },
    GetRow {
        table: u8,
        id: Id,
    },
    Select {
        table: u8,
        filter: Filt,
        limit: Size,
    },
    DeleteRows {
        table: u8,
        ids: Vec<Id>,
    },
    DefineIndex {
        index: u8,
        dim: u8,
        cosine: bool,
        m: Size,
        ef: Size,
    },
    UpsertVectors {
        index: u8,
        items: Vec<(Id, Vector)>,
    },
    /// Enough vectors at once to leave the exact-scan range and use the graph.
    BulkVectors {
        index: u8,
        count: u16,
        seed: u8,
    },
    Knn {
        index: u8,
        query: Vector,
        k: Size,
        ef: Size,
    },
    KnnFiltered {
        index: u8,
        table: u8,
        filter: Filt,
        query: Vector,
        k: Size,
        ef: Size,
    },
    UpsertEdges {
        edges: Vec<(Id, u8, Id, f32)>,
    },
    Neighbors {
        node: Id,
        rel: Option<u8>,
        dir: u8,
        limit: Size,
    },
    KHop {
        start: Id,
        rel: Option<u8>,
        dir: u8,
        depth: u32,
        limit: Size,
    },
    Counts,
    Flush,
    Reopen {
        sync_each_write: bool,
    },
    ReadOnly,
}

fn direction(d: u8) -> Direction {
    match d % 3 {
        0 => Direction::Out,
        1 => Direction::In,
        _ => Direction::Both,
    }
}

fn check_hits(hits: &[Hit], k: usize) {
    assert!(hits.len() <= k, "{} hits for k={k}", hits.len());
    assert!(
        hits.windows(2)
            .all(|w| w[0].distance.total_cmp(&w[1].distance).is_le()),
        "hits not sorted by distance: {hits:?}"
    );
    let mut ids: Vec<_> = hits.iter().map(|h| h.id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), hits.len(), "duplicate ids in hits: {hits:?}");
}

/// Dimension of each defined index, so most generated vectors are well formed.
struct Model {
    dims: [Option<usize>; 3],
}

fn apply(s: &mut NativeStore, model: &mut Model, op: &Op) {
    match op {
        Op::DefineTable { table, columns } => {
            let def = TableDef {
                name: pick(&TABLES, *table).into(),
                columns: columns
                    .iter()
                    .take(6)
                    .map(|(name, ty, indexed)| ColumnDef {
                        name: pick(&COLUMNS, *name).into(),
                        ty: [
                            ColumnType::Int,
                            ColumnType::Float,
                            ColumnType::Text,
                            ColumnType::Bytes,
                        ][*ty as usize % 4],
                        indexed: *indexed,
                    })
                    .collect(),
            };
            let _ = s.define_table(&def);
        }
        Op::UpsertRows { table, rows } => {
            let rows: Vec<_> = rows
                .iter()
                .take(64)
                .map(|(id, values)| Row {
                    id: id.get(),
                    values: values.iter().take(8).map(Val::get).collect(),
                })
                .collect();
            let _ = s.upsert_rows(pick(&TABLES, *table), &rows);
        }
        Op::GetRow { table, id } => {
            if let Ok(Some(row)) = s.get_row(pick(&TABLES, *table), id.get()) {
                assert_eq!(row.id, id.get());
            }
        }
        Op::Select {
            table,
            filter,
            limit,
        } => {
            let limit = limit.get();
            if let Ok(rows) = s.select(pick(&TABLES, *table), &filter.get(), limit) {
                assert!(rows.len() <= limit);
            }
        }
        Op::DeleteRows { table, ids } => {
            let ids: Vec<_> = ids.iter().take(64).map(Id::get).collect();
            let _ = s.delete_rows(pick(&TABLES, *table), &ids);
        }
        Op::DefineIndex {
            index,
            dim,
            cosine,
            m,
            ef,
        } => {
            let slot = *index as usize % INDEXES.len();
            let def = VectorIndexDef {
                name: INDEXES[slot].into(),
                dim: usize::from(*dim % 9),
                metric: if *cosine { Metric::Cosine } else { Metric::L2 },
                m: m.get(),
                ef_construction: ef.get(),
            };
            if s.define_vector_index(&def).is_ok() {
                model.dims[slot] = Some(def.dim);
            }
        }
        Op::UpsertVectors { index, items } => {
            let slot = *index as usize % INDEXES.len();
            let dim = model.dims[slot].unwrap_or(4);
            let vectors: Vec<_> = items
                .iter()
                .take(64)
                .map(|(id, v)| (id.get(), v.get(dim)))
                .collect();
            let items: Vec<_> = vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
            let _ = s.upsert_vectors(INDEXES[slot], &items);
        }
        Op::BulkVectors { index, count, seed } => {
            let slot = *index as usize % INDEXES.len();
            let Some(dim) = model.dims[slot] else {
                return;
            };
            let count = u64::from(*count % 1200);
            let vectors: Vec<Vec<f32>> = (0..count)
                .map(|i| {
                    (0..dim)
                        .map(|d| ((i * 31 + d as u64 * 17 + u64::from(*seed)) % 97) as f32)
                        .collect()
                })
                .collect();
            let items: Vec<_> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (1000 + i as u64, v.as_slice()))
                .collect();
            let _ = s.upsert_vectors(INDEXES[slot], &items);
        }
        Op::Knn {
            index,
            query,
            k,
            ef,
        } => {
            let slot = *index as usize % INDEXES.len();
            let q = query.get(model.dims[slot].unwrap_or(4));
            let k = k.get();
            if let Ok(hits) = s.knn(INDEXES[slot], &q, k, ef.get()) {
                check_hits(&hits, k);
            }
        }
        Op::KnnFiltered {
            index,
            table,
            filter,
            query,
            k,
            ef,
        } => {
            let slot = *index as usize % INDEXES.len();
            let q = query.get(model.dims[slot].unwrap_or(4));
            let k = k.get();
            if let Ok(hits) = s.knn_filtered(
                INDEXES[slot],
                &q,
                k,
                ef.get(),
                pick(&TABLES, *table),
                &filter.get(),
            ) {
                check_hits(&hits, k);
            }
        }
        Op::UpsertEdges { edges } => {
            let edges: Vec<_> = edges
                .iter()
                .take(64)
                .map(|(src, rel, dst, weight)| Edge {
                    src: src.get(),
                    rel: pick(&RELS, *rel).into(),
                    dst: dst.get(),
                    weight: *weight,
                })
                .collect();
            let _ = s.upsert_edges(&edges);
        }
        Op::Neighbors {
            node,
            rel,
            dir,
            limit,
        } => {
            let limit = limit.get();
            let rel = rel.map(|r| pick(&RELS, r));
            if let Ok(edges) = s.neighbors(node.get(), rel, direction(*dir), limit) {
                assert!(edges.len() <= limit);
            }
        }
        Op::KHop {
            start,
            rel,
            dir,
            depth,
            limit,
        } => {
            let limit = limit.get();
            let rel = rel.map(|r| pick(&RELS, r));
            if let Ok(reached) = s.k_hop(start.get(), rel, direction(*dir), *depth, limit) {
                assert!(
                    reached.len() <= limit,
                    "{} results for limit {limit}",
                    reached.len()
                );
                assert!(reached.iter().all(|r| r.id != start.get() && r.depth >= 1));
                assert!(reached.iter().all(|r| r.depth <= *depth));
                let mut ids: Vec<_> = reached.iter().map(|r| r.id).collect();
                ids.sort_unstable();
                ids.dedup();
                assert_eq!(ids.len(), reached.len(), "duplicate ids from k_hop");
            }
        }
        Op::Counts => {
            for table in TABLES {
                let _ = s.count_rows(table);
            }
            for index in INDEXES {
                let _ = s.count_vectors(index);
            }
            let _ = s.count_edges();
        }
        Op::Flush | Op::Reopen { .. } | Op::ReadOnly => {}
    }
}

fuzz_target!(|ops: Vec<Op>| {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut model = Model { dims: [None; 3] };
    let mut opts = OpenOptions::default();
    let mut store = Some(NativeStore::open(dir.path(), &opts).expect("create store"));
    for op in ops.iter().take(48) {
        match op {
            Op::Flush => {
                let _ = store.as_mut().expect("open store").flush();
            }
            Op::Reopen { sync_each_write } => {
                drop(store.take());
                opts.sync_each_write = *sync_each_write;
                store = Some(NativeStore::open(dir.path(), &opts).expect("reopen"));
            }
            Op::ReadOnly => {
                // A reader next to the live writer sees committed state and rejects writes.
                let mut reader = NativeStore::open_read_only(dir.path()).expect("read-only open");
                assert!(reader.flush().is_err(), "read-only flush must fail");
                for index in INDEXES {
                    let _ = reader.count_vectors(index);
                }
            }
            op => apply(store.as_mut().expect("open store"), &mut model, op),
        }
    }
    drop(store.take());
    let s = NativeStore::open(dir.path(), &opts).expect("final reopen");
    for index in INDEXES {
        let _ = s.count_vectors(index);
    }
});
