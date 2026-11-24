//! Conformance suite: the executable meaning of the [`Store`] contract.
//!
//! A backend calls `conformance::run_all::<MyStore>(empty_dir)` from one of its tests.
//! The suite is written by the planner, not by the backend lanes, so that a backend
//! is not the only judge of its own result.

#![allow(
    clippy::manual_is_multiple_of,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::type_complexity
)]

use crate::*;
use std::path::Path;

type Check = std::result::Result<(), String>;

macro_rules! ensure {
    ($cond:expr, $($arg:tt)*) => {
        if !$cond {
            return Err(format!($($arg)*));
        }
    };
}

fn e<T>(r: Result<T>, what: &str) -> std::result::Result<T, String> {
    r.map_err(|err| format!("{what}: {err}"))
}

/// Deterministic pseudo-random generator (64-bit linear congruential), so that every backend
/// is tested on the same data. The field is the state: equal seeds give equal sequences.
pub struct Lcg(pub u64);
impl Lcg {
    /// Next 31-bit value, in `0..2^31`.
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Next `f32` in `[-1, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
    }
    /// A vector of `dim` values from [`Lcg::next_f32`].
    pub fn vector(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next_f32()).collect()
    }
}

/// The table the suite uses: `docs` with columns `kind` (indexed text), `title` (text),
/// `size` (indexed int), `score` (float) and `blob` (bytes). Use it to run your own checks on the
/// same shape.
pub fn docs_table() -> TableDef {
    TableDef {
        name: "docs".to_string(),
        columns: vec![
            ColumnDef {
                name: "kind".to_string(),
                ty: ColumnType::Text,
                indexed: true,
            },
            ColumnDef {
                name: "title".to_string(),
                ty: ColumnType::Text,
                indexed: false,
            },
            ColumnDef {
                name: "size".to_string(),
                ty: ColumnType::Int,
                indexed: true,
            },
            ColumnDef {
                name: "score".to_string(),
                ty: ColumnType::Float,
                indexed: false,
            },
            ColumnDef {
                name: "blob".to_string(),
                ty: ColumnType::Bytes,
                indexed: false,
            },
        ],
    }
}

fn doc_row(id: NodeId) -> Row {
    let kind = if id % 3 == 0 { "paper" } else { "note" };
    Row {
        id,
        values: vec![
            Value::Text(kind.to_string()),
            Value::Text(format!("t{id:04} عنوان")),
            Value::Int(id as i64 * 10),
            if id % 5 == 0 {
                Value::Null
            } else {
                Value::Float(id as f64 / 4.0)
            },
            Value::Bytes(vec![(id % 251) as u8; (id % 7) as usize]),
        ],
    }
}

/// Reference distance between `a` and `b` under `metric`, in `f32`, as the contract defines it:
/// squared Euclidean for `L2`, `1 - cosine similarity` for `Cosine` (1 when either norm is zero).
/// The slices should have the same length; the extra components of the longer one are ignored.
pub fn exact_distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::L2 => a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum(),
        Metric::Cosine => {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na == 0.0 || nb == 0.0 {
                1.0
            } else {
                1.0 - dot / (na * nb)
            }
        }
    }
}

fn open<S: Store>(dir: &Path) -> std::result::Result<S, String> {
    e(S::open(dir, &OpenOptions::default()), "open")
}

/// Rows: upsert, get, replace, select with every filter, count, delete, schema errors.
pub fn relational<S: Store>(dir: &Path) -> Check {
    let mut s: S = open(dir)?;
    e(s.define_table(&docs_table()), "define_table")?;
    e(
        s.define_table(&docs_table()),
        "define_table twice (identical) must be a no-op",
    )?;
    let mut other = docs_table();
    other.columns.pop();
    ensure!(
        matches!(s.define_table(&other), Err(Error::Schema(_))),
        "redefinition with a different shape must be Error::Schema"
    );
    let bad = TableDef {
        name: "Bad Name".to_string(),
        columns: vec![],
    };
    ensure!(
        matches!(s.define_table(&bad), Err(Error::Schema(_))),
        "illegal table name must be Error::Schema"
    );

    let rows: Vec<Row> = (1..=100).rev().map(doc_row).collect();
    e(s.upsert_rows("docs", &rows), "upsert_rows")?;
    ensure!(
        e(s.count_rows("docs"), "count")? == 100,
        "count after insert must be 100"
    );
    ensure!(
        e(s.get_row("docs", 42), "get")? == Some(doc_row(42)),
        "get_row(42) must round-trip every value type, including Arabic text, Null and Bytes"
    );
    ensure!(
        e(s.get_row("docs", 1000), "get missing")?.is_none(),
        "missing id must be None"
    );
    ensure!(
        matches!(s.get_row("nope", 1), Err(Error::NotFound(_))),
        "unknown table must be Error::NotFound"
    );

    let mut changed = doc_row(42);
    changed.values[1] = Value::Text("replaced".to_string());
    e(s.upsert_rows("docs", &[changed.clone()]), "upsert replace")?;
    ensure!(
        e(s.count_rows("docs"), "count")? == 100,
        "replace must not add a row"
    );
    ensure!(
        e(s.get_row("docs", 42), "get")? == Some(changed),
        "replace must overwrite"
    );

    let short = Row {
        id: 500,
        values: vec![Value::Null],
    };
    ensure!(
        matches!(s.upsert_rows("docs", &[short]), Err(Error::Schema(_))),
        "wrong value count must be Error::Schema"
    );
    let mut wrong = doc_row(501);
    wrong.values[2] = Value::Text("not an int".to_string());
    ensure!(
        matches!(
            s.upsert_rows("docs", &[doc_row(502), wrong]),
            Err(Error::Schema(_))
        ),
        "wrong value type must be Error::Schema"
    );
    ensure!(
        e(s.get_row("docs", 502), "get")?.is_none(),
        "a failed upsert_rows must be atomic (row 502 must not exist)"
    );

    let all = e(s.select("docs", &Filter::All, 1000), "select all")?;
    ensure!(
        all.len() == 100 && all.windows(2).all(|w| w[0].id < w[1].id),
        "select All must return 100 rows in ascending id"
    );
    ensure!(
        e(s.select("docs", &Filter::All, 7), "select limit")? == all[..7],
        "select limit must return the first 7 rows in full order"
    );
    let papers = e(
        s.select(
            "docs",
            &Filter::Eq {
                column: "kind".to_string(),
                value: Value::Text("paper".to_string()),
            },
            1000,
        ),
        "select eq",
    )?;
    ensure!(
        papers.len() == 33 && papers.iter().all(|r| r.id % 3 == 0),
        "Eq on kind must return the 33 papers, got {}",
        papers.len()
    );
    ensure!(
        papers.windows(2).all(|w| w[0].id < w[1].id),
        "select Eq must return ascending ids: {papers:?}"
    );
    let paper_filter = Filter::Eq {
        column: "kind".to_string(),
        value: Value::Text("paper".to_string()),
    };
    ensure!(
        e(s.select("docs", &paper_filter, 7), "select Eq limit")? == papers[..7],
        "select Eq limit must return the first 7 papers"
    );
    let range = e(
        s.select(
            "docs",
            &Filter::IntRange {
                column: "size".to_string(),
                min: 100,
                max: 200,
            },
            1000,
        ),
        "select range",
    )?;
    ensure!(
        range.iter().map(|r| r.id).collect::<Vec<_>>() == (10..=20).collect::<Vec<_>>(),
        "IntRange 100..=200 must return ids 10..=20"
    );
    let prefix = e(
        s.select(
            "docs",
            &Filter::TextPrefix {
                column: "title".to_string(),
                prefix: "t001".to_string(),
            },
            1000,
        ),
        "select prefix",
    )?;
    ensure!(
        prefix.iter().map(|r| r.id).collect::<Vec<_>>() == (10..=19).collect::<Vec<_>>(),
        "TextPrefix t001 must return ids 10..=19"
    );
    ensure!(
        matches!(
            s.select(
                "docs",
                &Filter::Eq {
                    column: "nope".to_string(),
                    value: Value::Int(1)
                },
                10
            ),
            Err(Error::NotFound(_))
        ),
        "unknown column must be Error::NotFound"
    );

    ensure!(
        e(s.delete_rows("docs", &[1, 2, 9999]), "delete")? == 2,
        "delete must report 2 existing rows"
    );
    ensure!(
        e(s.count_rows("docs"), "count")? == 98,
        "count after delete must be 98"
    );
    let remaining = e(
        s.select(
            "docs",
            &Filter::Eq {
                column: "kind".to_string(),
                value: Value::Text("note".to_string()),
            },
            1000,
        ),
        "indexed select after delete",
    )?;
    ensure!(
        remaining.len() == 65 && remaining.iter().all(|r| r.id != 1 && r.id != 2),
        "indexed select must exclude deleted ids 1 and 2: {remaining:?}"
    );
    Ok(())
}

/// Vectors: exactness on a small set, both metrics, replace, filtered search, errors.
pub fn vectors<S: Store>(dir: &Path) -> Check {
    let mut s: S = open(dir)?;
    e(s.define_table(&docs_table()), "define_table")?;
    let rows: Vec<Row> = (1..=300).map(doc_row).collect();
    e(s.upsert_rows("docs", &rows), "upsert_rows")?;
    for (name, metric) in [("emb_cos", Metric::Cosine), ("emb_l2", Metric::L2)] {
        let def = VectorIndexDef {
            name: name.to_string(),
            dim: 16,
            metric,
            m: 16,
            ef_construction: 200,
        };
        e(s.define_vector_index(&def), "define_vector_index")?;
        e(
            s.define_vector_index(&def),
            "define_vector_index twice (identical) must be a no-op",
        )?;
        let mut rng = Lcg(7);
        let data: Vec<(NodeId, Vec<f32>)> = (1..=300u64).map(|id| (id, rng.vector(16))).collect();
        let items: Vec<(NodeId, &[f32])> = data.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        for chunk in items.chunks(64) {
            e(s.upsert_vectors(name, chunk), "upsert_vectors")?;
        }
        ensure!(
            e(s.count_vectors(name), "count_vectors")? == 300,
            "{name}: count_vectors must be 300"
        );
        ensure!(
            matches!(
                s.upsert_vectors(name, &[(999, &[0.5f32; 3][..])]),
                Err(Error::Schema(_))
            ),
            "{name}: wrong dimension must be Error::Schema"
        );

        // With 300 points and ef_search 300 an HNSW index must be exact or very nearly so.
        let mut found = 0usize;
        let mut wanted = 0usize;
        for _ in 0..20 {
            let q = rng.vector(16);
            let mut truth: Vec<(f32, NodeId)> = data
                .iter()
                .map(|(id, v)| (exact_distance(metric, &q, v), *id))
                .collect();
            truth.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let hits = e(s.knn(name, &q, 10, 300), "knn")?;
            ensure!(
                hits.len() == 10,
                "{name}: knn must return 10 hits, got {}",
                hits.len()
            );
            ensure!(
                hits.windows(2).all(|w| w[0].distance <= w[1].distance),
                "{name}: hits must be in ascending distance"
            );
            for h in &hits {
                let (_, v) = data
                    .iter()
                    .find(|(id, _)| *id == h.id)
                    .ok_or_else(|| format!("{name}: unknown id {}", h.id))?;
                let d = exact_distance(metric, &q, v);
                ensure!(
                    (d - h.distance).abs() <= 1e-3 * (1.0 + d.abs()),
                    "{name}: reported distance {} differs from the contract's {} for id {}",
                    h.distance,
                    d,
                    h.id
                );
            }
            wanted += 10;
            found += truth[..10]
                .iter()
                .filter(|(_, id)| hits.iter().any(|h| h.id == *id))
                .count();

            let f = Filter::Eq {
                column: "kind".to_string(),
                value: Value::Text("paper".to_string()),
            };
            let fh = e(s.knn_filtered(name, &q, 5, 300, "docs", &f), "knn_filtered")?;
            ensure!(
                fh.len() == 5 && fh.iter().all(|h| h.id % 3 == 0),
                "{name}: knn_filtered must return 5 hits that are all papers"
            );
            ensure!(
                fh.windows(2).all(|w| w[0].distance < w[1].distance
                    || (w[0].distance == w[1].distance && w[0].id < w[1].id)),
                "{name}: knn_filtered must be ordered by (distance, id): {fh:?}"
            );
            let best_paper = truth.iter().find(|(_, id)| id % 3 == 0).map(|(_, id)| *id);
            ensure!(
                fh.iter().any(|h| Some(h.id) == best_paper),
                "{name}: knn_filtered must contain the nearest paper"
            );
        }
        ensure!(
            found * 100 >= wanted * 95,
            "{name}: recall@10 at ef_search=300 on 300 points is {found}/{wanted}, need >= 95%"
        );

        // Replace: move vector 1 onto a far point, then it must be its own nearest neighbour there.
        let far = vec![50.0f32; 16];
        let mut far1 = far.clone();
        far1[0] = 49.0;
        e(
            s.upsert_vectors(name, &[(1, far1.as_slice())]),
            "replace vector",
        )?;
        ensure!(
            e(s.count_vectors(name), "count_vectors")? == 300,
            "{name}: replace must not add a vector"
        );
        let hits = e(
            s.knn(name, &far1, 1, 300),
            "knn after replace at new position",
        )?;
        ensure!(
            hits.len() == 1 && hits[0].id == 1 && hits[0].distance.abs() < 1e-4,
            "{name}: after replace, id 1 must be nearest to its new position: {hits:?}"
        );
        // Replace-then-search: at its old position, id 1 may only appear with the distance of its new vector.
        let old = data[0].1.clone();
        let want = exact_distance(metric, &old, &far1);
        let hits = e(
            s.knn(name, &old, 5, 300),
            "knn at the replaced vector's old position",
        )?;
        ensure!(
            hits.iter().all(|h| h.id != 1 || (h.distance - want).abs() <= 1e-3 * (1.0 + want.abs())),
            "{name}: after replace, a search at the old position still answers from the old vector of id 1"
        );
        // Non-finite input is refused and changes nothing.
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bad = vec![0.25f32; 16];
            bad[3] = poison;
            ensure!(
                matches!(
                    s.upsert_vectors(name, &[(998, bad.as_slice())]),
                    Err(Error::Schema(_))
                ),
                "{name}: a vector with a non-finite value must be Error::Schema"
            );
        }
        ensure!(
            e(s.count_vectors(name), "count_vectors")? == 300,
            "{name}: a refused vector must not be stored"
        );
    }
    ensure!(
        matches!(s.knn("nope", &[0.0; 16], 1, 10), Err(Error::NotFound(_))),
        "unknown index must be Error::NotFound"
    );
    Ok(())
}

/// Graph: upsert, neighbours in each direction, BFS depth and order, cycles, relation filter.
pub fn graph<S: Store>(dir: &Path) -> Check {
    let mut s: S = open(dir)?;
    let ed = |src: NodeId, rel: &str, dst: NodeId| Edge {
        src,
        rel: rel.to_string(),
        dst,
        weight: 1.0,
    };
    // chain 1->2->3->4->5 (cites), cycle 5->1 (cites), branch 2->10, 2->11 (links), self loop 3->3 (links)
    let edges = vec![
        ed(3, "links", 3),
        ed(2, "links", 11),
        ed(2, "links", 10),
        ed(5, "cites", 1),
        ed(4, "cites", 5),
        ed(3, "cites", 4),
        ed(2, "cites", 3),
        ed(1, "cites", 2),
    ];
    e(s.upsert_edges(&edges), "upsert_edges")?;
    e(
        s.upsert_edges(&[Edge {
            src: 1,
            rel: "cites".to_string(),
            dst: 2,
            weight: 0.25,
        }]),
        "upsert same edge",
    )?;
    ensure!(
        e(s.count_edges(), "count_edges")? == 8,
        "the same (src, rel, dst) must replace, not duplicate"
    );

    let out2 = e(s.neighbors(2, None, Direction::Out, 100), "neighbors out")?;
    ensure!(
        out2 == vec![ed(2, "cites", 3), ed(2, "links", 10), ed(2, "links", 11)],
        "neighbors(2, Out) must return full edges ordered by (rel, src, dst): {out2:?}"
    );
    let in2 = e(s.neighbors(2, None, Direction::In, 100), "neighbors in")?;
    ensure!(
        in2 == vec![Edge {
            src: 1,
            rel: "cites".to_string(),
            dst: 2,
            weight: 0.25
        }],
        "neighbors(2, In) must be the full replaced edge with weight 0.25: {in2:?}"
    );
    ensure!(
        e(
            s.neighbors(2, Some("links"), Direction::Both, 100),
            "neighbors rel"
        )?
        .len()
            == 2,
        "relation filter must keep only links"
    );
    ensure!(
        e(s.neighbors(3, None, Direction::Both, 100), "neighbors both")?.len() == 3,
        "Both on node 3 must return 3 edges (self-loop once)"
    );
    ensure!(
        e(s.neighbors(2, None, Direction::Out, 2), "neighbors limit")? == out2[..2],
        "neighbors limit must return first 2 full edges"
    );
    ensure!(
        e(
            s.neighbors(777, None, Direction::Both, 10),
            "neighbors unknown"
        )?
        .is_empty(),
        "unknown node has no neighbours"
    );

    let r = |id: NodeId, depth: u32| Reached { id, depth };
    let hop2 = e(s.k_hop(1, None, Direction::Out, 2, 100), "k_hop")?;
    ensure!(
        hop2 == vec![r(2, 1), r(3, 2), r(10, 2), r(11, 2)],
        "k_hop(1, Out, 2) wrong: {hop2:?}"
    );
    let all = e(
        s.k_hop(1, Some("cites"), Direction::Out, 10, 100),
        "k_hop cycle",
    )?;
    ensure!(
        all == vec![r(2, 1), r(3, 2), r(4, 3), r(5, 4)],
        "k_hop over a cycle must stop, exclude the start and keep the smallest depth: {all:?}"
    );
    let back = e(s.k_hop(1, Some("cites"), Direction::In, 2, 100), "k_hop in")?;
    ensure!(
        back == vec![r(5, 1), r(4, 2)],
        "k_hop(1, In, 2) wrong: {back:?}"
    );
    let both = e(s.k_hop(1, None, Direction::Both, 1, 100), "k_hop both")?;
    ensure!(
        both == vec![r(2, 1), r(5, 1)],
        "k_hop(1, Both, 1) wrong: {both:?}"
    );
    let full = e(
        s.k_hop(1, None, Direction::Out, 10, 100),
        "k_hop full order",
    )?;
    ensure!(
        full == vec![r(2, 1), r(3, 2), r(10, 2), r(11, 2), r(4, 3), r(5, 4)],
        "k_hop full order wrong: {full:?}"
    );
    ensure!(
        e(s.k_hop(1, None, Direction::Out, 10, 3), "k_hop limit")? == full[..3],
        "k_hop limit must return first 3 reached nodes"
    );
    ensure!(
        e(s.k_hop(1, None, Direction::Out, 0, 100), "k_hop depth 0")?.is_empty(),
        "max_depth 0 reaches nothing"
    );
    Ok(())
}

/// Durability: everything written before `flush()` is there after drop and reopen.
pub fn persistence<S: Store>(dir: &Path) -> Check {
    let def = VectorIndexDef {
        name: "emb".to_string(),
        dim: 8,
        metric: Metric::L2,
        m: 16,
        ef_construction: 100,
    };
    let mut rng = Lcg(11);
    let data: Vec<(NodeId, Vec<f32>)> = (1..=200u64).map(|id| (id, rng.vector(8))).collect();
    {
        let mut s: S = open(dir)?;
        e(s.define_table(&docs_table()), "define_table")?;
        e(
            s.upsert_rows("docs", &(1..=200).map(doc_row).collect::<Vec<_>>()),
            "upsert_rows",
        )?;
        e(s.define_vector_index(&def), "define_vector_index")?;
        let items: Vec<(NodeId, &[f32])> = data.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        e(s.upsert_vectors("emb", &items), "upsert_vectors")?;
        e(
            s.upsert_edges(
                &(1..200u64)
                    .map(|i| Edge {
                        src: i,
                        rel: "next".to_string(),
                        dst: i + 1,
                        weight: 1.0,
                    })
                    .collect::<Vec<_>>(),
            ),
            "upsert_edges",
        )?;
        e(s.flush(), "flush")?;
    }
    ensure!(
        e(dir_size(dir).map_err(Error::Io), "dir_size")? > 0,
        "a flushed store must have files in its directory"
    );
    let missing = dir.join("does-not-exist");
    ensure!(
        S::open(
            &missing,
            &OpenOptions {
                create: false,
                sync_each_write: false
            }
        )
        .is_err(),
        "open with create=false on a missing directory must fail"
    );
    let mut s: S = e(
        S::open(
            dir,
            &OpenOptions {
                create: false,
                sync_each_write: true,
            },
        ),
        "reopen",
    )?;
    ensure!(
        e(s.count_rows("docs"), "count_rows")? == 200,
        "rows must survive reopen"
    );
    ensure!(
        e(s.get_row("docs", 150), "get_row")? == Some(doc_row(150)),
        "row 150 must survive reopen unchanged"
    );
    ensure!(
        e(s.count_vectors("emb"), "count_vectors")? == 200,
        "vectors must survive reopen"
    );
    ensure!(
        e(s.count_edges(), "count_edges")? == 199,
        "edges must survive reopen"
    );
    let hits = e(s.knn("emb", &data[76].1, 1, 200), "knn after reopen")?;
    ensure!(
        hits.len() == 1 && hits[0].id == 77 && hits[0].distance.abs() < 1e-4,
        "the vector index must answer after reopen (query = vector 77)"
    );
    let reached = e(
        s.k_hop(1, Some("next"), Direction::Out, 5, 100),
        "k_hop after reopen",
    )?;
    ensure!(
        reached
            == (1..=5)
                .map(|depth| Reached {
                    id: u64::from(depth) + 1,
                    depth
                })
                .collect::<Vec<_>>(),
        "k_hop after reopen must preserve exact ids and depths: {reached:?}"
    );
    e(
        s.define_table(&docs_table()),
        "define_table after reopen must be a no-op",
    )?;
    e(
        s.define_vector_index(&def),
        "define_vector_index after reopen must be a no-op",
    )?;
    e(s.upsert_rows("docs", &[doc_row(201)]), "write after reopen")?;
    e(s.flush(), "flush after reopen")?;
    Ok(())
}

fn small_index(metric: Metric) -> VectorIndexDef {
    VectorIndexDef {
        name: "emb".to_string(),
        dim: 2,
        metric,
        m: 16,
        ef_construction: 100,
    }
}

fn schema_check<T: std::fmt::Debug>(result: Result<T>, input: &str, failures: &mut Vec<String>) {
    if !matches!(result, Err(Error::Schema(_))) {
        failures.push(format!("{input}: expected Error::Schema, got {result:?}"));
    }
}

fn finish_checks(failures: Vec<String>) -> Check {
    ensure!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

fn oversized_ids<S: Store>(dir: &Path) -> Check {
    let mut s: S = open(dir)?;
    e(s.define_table(&docs_table()), "define_table")?;
    e(
        s.define_vector_index(&small_index(Metric::L2)),
        "define_vector_index",
    )?;
    let mut failures = Vec::new();
    for id in [i64::MAX as u64 + 1, u64::MAX] {
        let row = Row {
            id,
            values: doc_row(1).values,
        };
        schema_check(
            s.upsert_rows("docs", &[row]),
            &format!("upsert_rows id={id}"),
            &mut failures,
        );
        schema_check(
            s.get_row("docs", id),
            &format!("get_row id={id}"),
            &mut failures,
        );
        schema_check(
            s.delete_rows("docs", &[id]),
            &format!("delete_rows id={id}"),
            &mut failures,
        );
        schema_check(
            s.upsert_vectors("emb", &[(id, &[1.0, 0.0])]),
            &format!("upsert_vectors id={id}"),
            &mut failures,
        );
        for (src, dst) in [(id, 1), (1, id)] {
            schema_check(
                s.upsert_edges(&[Edge {
                    src,
                    rel: "r".to_string(),
                    dst,
                    weight: 1.0,
                }]),
                &format!("upsert_edges src={src} dst={dst}"),
                &mut failures,
            );
        }
        for direction in [Direction::Out, Direction::In, Direction::Both] {
            schema_check(
                s.neighbors(id, None, direction, 10),
                &format!("neighbors node={id} dir={direction:?}"),
                &mut failures,
            );
            schema_check(
                s.neighbors(id, None, direction, 0),
                &format!("neighbors node={id} dir={direction:?} limit=0"),
                &mut failures,
            );
            schema_check(
                s.k_hop(id, None, direction, 2, 10),
                &format!("k_hop start={id} dir={direction:?}"),
                &mut failures,
            );
            schema_check(
                s.k_hop(id, None, direction, 0, 0),
                &format!("k_hop start={id} dir={direction:?} depth=0 limit=0"),
                &mut failures,
            );
        }
    }
    finish_checks(failures)
}

fn vector_queries<S: Store>(dir: &Path) -> Check {
    let mut s: S = open(dir)?;
    e(s.define_table(&docs_table()), "define_table")?;
    let mut failures = Vec::new();
    for metric in [Metric::Cosine, Metric::L2] {
        let mut def = small_index(metric);
        def.name = if metric == Metric::Cosine {
            "cos"
        } else {
            "l2"
        }
        .to_string();
        e(s.define_vector_index(&def), "define_vector_index")?;
        e(
            s.upsert_vectors(&def.name, &[(1, &[1.0, 0.0])]),
            "upsert_vectors",
        )?;
        for query in [
            vec![],
            vec![1.0],
            vec![1.0, 0.0, 0.0],
            vec![f32::NAN, 0.0],
            vec![f32::INFINITY, 0.0],
            vec![f32::NEG_INFINITY, 0.0],
        ] {
            for k in [0, 1] {
                schema_check(
                    s.knn(&def.name, &query, k, 100),
                    &format!("{metric:?} knn query={query:?} k={k}"),
                    &mut failures,
                );
                schema_check(
                    s.knn_filtered(&def.name, &query, k, 100, "docs", &Filter::All),
                    &format!("{metric:?} knn_filtered query={query:?} k={k}"),
                    &mut failures,
                );
            }
        }
        let mut changed = def.clone();
        changed.dim = 3;
        schema_check(
            s.define_vector_index(&changed),
            &format!("{metric:?} redefine dim=3"),
            &mut failures,
        );
        changed = def.clone();
        changed.metric = if metric == Metric::Cosine {
            Metric::L2
        } else {
            Metric::Cosine
        };
        schema_check(
            s.define_vector_index(&changed),
            &format!("{metric:?} redefine metric={:?}", changed.metric),
            &mut failures,
        );
        ensure!(
            e(
                s.count_vectors(&def.name),
                "count after rejected redefinition"
            )? == 1,
            "redefinition must leave vectors intact"
        );
        let hits = e(
            s.knn(&def.name, &[1.0, 0.0], 1, 100),
            "query after rejected redefinition",
        )?;
        ensure!(
            hits == vec![Hit {
                id: 1,
                distance: 0.0
            }],
            "redefinition must leave original dimension and metric intact: {hits:?}"
        );
    }
    finish_checks(failures)
}

fn vector_ties<S: Store>(dir: &Path, metric: Metric) -> Check {
    let mut s: S = open(dir)?;
    e(s.define_table(&docs_table()), "define_table")?;
    e(
        s.upsert_rows("docs", &[doc_row(9), doc_row(3), doc_row(6), doc_row(12)]),
        "upsert rows for ties",
    )?;
    e(
        s.define_vector_index(&small_index(metric)),
        "define_vector_index",
    )?;
    e(
        s.upsert_vectors(
            "emb",
            &[
                (9, &[1.0, 0.0]),
                (3, &[1.0, 0.0]),
                (6, &[1.0, 0.0]),
                (12, &[1.0, 0.0]),
            ],
        ),
        "insert identical vectors in non-id order",
    )?;
    let filter = Filter::Eq {
        column: "kind".to_string(),
        value: Value::Text("paper".to_string()),
    };
    let mut failures = Vec::new();
    for k in [1, 2, 3, 4, 5] {
        let expected: Vec<_> = [3, 6, 9, 12]
            .into_iter()
            .take(k)
            .map(|id| Hit { id, distance: 0.0 })
            .collect();
        let hits = e(s.knn("emb", &[1.0, 0.0], k, 100), "knn ties")?;
        if hits != expected {
            failures.push(format!("{metric:?} knn identical [1,0] ids=[9,3,6,12] k={k}: expected {expected:?}, got {hits:?}"));
        }
        let hits = e(
            s.knn_filtered("emb", &[1.0, 0.0], k, 100, "docs", &filter),
            "knn_filtered ties",
        )?;
        if hits != expected {
            failures.push(format!("{metric:?} knn_filtered identical [1,0] ids=[9,3,6,12] k={k}: expected {expected:?}, got {hits:?}"));
        }
    }
    let (query, farther, nearer) = if metric == Metric::L2 {
        ([0.0, 0.0], [1.0 + f32::EPSILON, 0.0], [1.0, 0.0])
    } else {
        ([0.0, 1.0], [1.0, 0.0], [1.0, f32::EPSILON])
    };
    e(
        s.upsert_vectors(
            "emb",
            &[
                (3, &farther),
                (6, &nearer),
                (9, &[0.0, -4.0]),
                (12, &[0.0, -4.0]),
            ],
        ),
        "replace with near but unequal distances",
    )?;
    for filtered in [false, true] {
        let hits = if filtered {
            e(
                s.knn_filtered("emb", &query, 2, 100, "docs", &filter),
                "knn_filtered near ties",
            )?
        } else {
            e(s.knn("emb", &query, 2, 100), "knn near ties")?
        };
        if hits.len() != 2
            || hits[0].id != 6
            || hits[1].id != 3
            || hits[0].distance >= hits[1].distance
        {
            failures.push(format!("{metric:?} near unequal distances query={query:?} id3={farther:?} id6={nearer:?} filtered={filtered}: expected closer id6 before id3, got {hits:?}"));
        }
    }
    finish_checks(failures)
}

fn zero_limits<S: Store>(dir: &Path) -> Check {
    let mut s: S = open(dir)?;
    e(s.define_table(&docs_table()), "define_table")?;
    e(s.upsert_rows("docs", &[doc_row(1)]), "upsert_rows")?;
    e(
        s.upsert_edges(&[Edge {
            src: 1,
            rel: "r".to_string(),
            dst: 2,
            weight: 1.0,
        }]),
        "upsert_edges",
    )?;
    for filter in [
        Filter::All,
        Filter::Eq {
            column: "kind".to_string(),
            value: Value::Text("note".to_string()),
        },
    ] {
        ensure!(
            e(s.select("docs", &filter, 0), "select limit=0")?.is_empty(),
            "select limit=0 must be empty for {filter:?}"
        );
    }
    for dir in [Direction::Out, Direction::In, Direction::Both] {
        ensure!(
            e(s.neighbors(1, None, dir, 0), "neighbors limit=0")?.is_empty(),
            "neighbors limit=0 must be empty for {dir:?}"
        );
        ensure!(
            e(s.k_hop(1, None, dir, 2, 0), "k_hop limit=0")?.is_empty(),
            "k_hop limit=0 must be empty for {dir:?}"
        );
    }
    for metric in [Metric::Cosine, Metric::L2] {
        let mut def = small_index(metric);
        def.name = if metric == Metric::Cosine {
            "cos"
        } else {
            "l2"
        }
        .to_string();
        e(s.define_vector_index(&def), "define_vector_index")?;
        e(
            s.upsert_vectors(&def.name, &[(1, &[1.0, 0.0])]),
            "upsert_vectors",
        )?;
        ensure!(
            e(s.knn(&def.name, &[1.0, 0.0], 0, 100), "knn k=0")?.is_empty(),
            "{metric:?} knn k=0 must be empty"
        );
        ensure!(
            e(
                s.knn_filtered(&def.name, &[1.0, 0.0], 0, 100, "docs", &Filter::All),
                "knn_filtered k=0"
            )?
            .is_empty(),
            "{metric:?} knn_filtered k=0 must be empty"
        );
    }
    Ok(())
}

fn cosine_zero<S: Store>(dir: &Path) -> Check {
    let mut s: S = open(dir)?;
    e(
        s.define_vector_index(&small_index(Metric::Cosine)),
        "define_vector_index",
    )?;
    e(
        s.upsert_vectors(
            "emb",
            &[(3, &[0.0, 0.0]), (2, &[-1.0, 0.0]), (1, &[1.0, 0.0])],
        ),
        "upsert zero and opposite vectors",
    )?;
    let hits = e(s.knn("emb", &[1.0, 0.0], 3, 100), "cosine zero vector")?;
    ensure!(
        hits == vec![
            Hit {
                id: 1,
                distance: 0.0
            },
            Hit {
                id: 3,
                distance: 1.0
            },
            Hit {
                id: 2,
                distance: 2.0
            }
        ],
        "cosine zero-vector distance must be 1: {hits:?}"
    );
    let hits = e(s.knn("emb", &[0.0, 0.0], 2, 100), "cosine zero query")?;
    ensure!(
        hits == vec![
            Hit {
                id: 1,
                distance: 1.0
            },
            Hit {
                id: 2,
                distance: 1.0
            }
        ],
        "cosine zero-query distances must be 1 with id tie order: {hits:?}"
    );
    Ok(())
}

fn deletion_isolation<S: Store>(dir: &Path) -> Check {
    let mut s: S = open(dir)?;
    e(s.define_table(&docs_table()), "define_table")?;
    e(
        s.upsert_rows("docs", &[doc_row(3), doc_row(6), doc_row(9)]),
        "upsert_rows",
    )?;
    let edge = Edge {
        src: 3,
        rel: "r".to_string(),
        dst: 6,
        weight: 0.5,
    };
    e(s.upsert_edges(std::slice::from_ref(&edge)), "upsert_edges")?;
    for (name, metric) in [("cos", Metric::Cosine), ("l2", Metric::L2)] {
        let mut def = small_index(metric);
        def.name = name.to_string();
        e(s.define_vector_index(&def), "define_vector_index")?;
        e(
            s.upsert_vectors(
                name,
                &[(3, &[1.0, 0.0]), (6, &[0.0, 1.0]), (9, &[-1.0, 0.0])],
            ),
            "upsert_vectors",
        )?;
    }
    ensure!(
        e(
            s.delete_rows("docs", &[3, 3, 999]),
            "delete duplicate and missing ids"
        )? == 1,
        "delete [3,3,999] must count one distinct existing row"
    );
    ensure!(
        e(s.get_row("docs", 3), "get deleted row")?.is_none(),
        "deleted row 3 must be absent"
    );
    let filter = Filter::Eq {
        column: "kind".to_string(),
        value: Value::Text("paper".to_string()),
    };
    let rows = e(s.select("docs", &filter, 10), "indexed select after delete")?;
    ensure!(
        rows == vec![doc_row(6), doc_row(9)],
        "indexed select after delete must return only rows 6 and 9: {rows:?}"
    );
    for name in ["cos", "l2"] {
        ensure!(
            e(s.count_vectors(name), "count_vectors after row delete")? == 3,
            "{name}: row deletion must not delete vectors"
        );
        let hits = e(s.knn(name, &[1.0, 0.0], 1, 100), "knn after row delete")?;
        ensure!(
            hits == vec![Hit {
                id: 3,
                distance: 0.0
            }],
            "{name}: deleted row's vector must remain unchanged: {hits:?}"
        );
        for f in [&filter, &Filter::All] {
            let hits = e(
                s.knn_filtered(name, &[1.0, 0.0], 10, 100, "docs", f),
                "knn_filtered after row delete",
            )?;
            ensure!(
                hits.iter().map(|h| h.id).collect::<Vec<_>>() == vec![6, 9],
                "{name}: filtered search must exclude deleted row 3 and retain rows 6,9: {hits:?}"
            );
        }
    }
    ensure!(
        e(
            s.neighbors(3, None, Direction::Out, 10),
            "neighbors after row delete"
        )? == vec![edge],
        "row deletion must not change the shared-id edge"
    );
    Ok(())
}

fn batch_validation<S: Store>(dir: &Path) -> Check {
    let mut s: S = open(dir)?;
    e(s.define_table(&docs_table()), "define_table")?;
    e(s.upsert_rows("docs", &[doc_row(1)]), "upsert_rows")?;
    let mut failures = Vec::new();
    schema_check(
        s.delete_rows("docs", &[1, i64::MAX as u64 + 1]),
        "delete_rows [1,9223372036854775808]",
        &mut failures,
    );
    ensure!(
        e(s.get_row("docs", 1), "row after rejected delete")? == Some(doc_row(1)),
        "rejected delete batch must leave row 1 unchanged"
    );
    e(
        s.define_vector_index(&small_index(Metric::L2)),
        "define_vector_index",
    )?;
    e(
        s.upsert_vectors("emb", &[(1, &[1.0, 0.0])]),
        "upsert_vectors",
    )?;
    for bad in [vec![1.0], vec![f32::NAN, 0.0], vec![f32::INFINITY, 0.0]] {
        schema_check(
            s.upsert_vectors("emb", &[(1, &[2.0, 0.0]), (2, &bad)]),
            &format!("upsert_vectors [(1,[2,0]),(2,{bad:?})]"),
            &mut failures,
        );
        ensure!(
            e(s.count_vectors("emb"), "count after rejected batch")? == 1,
            "rejected vector batch must not insert id 2"
        );
        let hits = e(
            s.knn("emb", &[1.0, 0.0], 1, 100),
            "knn after rejected batch",
        )?;
        ensure!(
            hits == vec![Hit {
                id: 1,
                distance: 0.0
            }],
            "rejected vector batch must not replace id 1: {hits:?}"
        );
    }
    let original = Edge {
        src: 1,
        rel: "Any relation!".to_string(),
        dst: 2,
        weight: 0.25,
    };
    e(
        s.upsert_edges(&[
            original.clone(),
            Edge {
                weight: 0.5,
                ..original.clone()
            },
        ]),
        "duplicate edge and unrestricted relation",
    )?;
    ensure!(
        e(
            s.neighbors(1, Some("Any relation!"), Direction::Out, 10),
            "duplicate edge result"
        )? == vec![Edge {
            weight: 0.5,
            ..original.clone()
        }],
        "later duplicate edge must win and relation strings need not be valid_name"
    );
    for weight in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let bad = Edge {
            src: 3,
            rel: "r".to_string(),
            dst: 4,
            weight,
        };
        schema_check(
            s.upsert_edges(&[original.clone(), bad]),
            &format!("upsert_edges [(1,Any relation!,2,0.25),(3,r,4,{weight})]"),
            &mut failures,
        );
        ensure!(
            e(s.count_edges(), "edge count after rejected batch")? == 1,
            "rejected edge batch must not insert an edge"
        );
        ensure!(
            e(
                s.neighbors(1, None, Direction::Out, 10),
                "edge after rejected batch"
            )? == vec![Edge {
                weight: 0.5,
                ..original.clone()
            }],
            "rejected edge batch must not replace an edge"
        );
    }
    finish_checks(failures)
}

/// Run every check, each in its own sub-directory of the empty directory `base`.
/// Returns the failures as `(check name, message)`; empty means the backend conforms.
pub fn run_all<S: Store>(base: &Path) -> Vec<(&'static str, String)> {
    let checks: [(&'static str, fn(&Path) -> Check); 12] = [
        ("relational", relational::<S>),
        ("vectors", vectors::<S>),
        ("graph", graph::<S>),
        ("persistence", persistence::<S>),
        ("oversized_ids", oversized_ids::<S>),
        ("vector_queries", vector_queries::<S>),
        ("vector_ties_l2", |dir| vector_ties::<S>(dir, Metric::L2)),
        ("vector_ties_cosine", |dir| {
            vector_ties::<S>(dir, Metric::Cosine)
        }),
        ("zero_limits", zero_limits::<S>),
        ("cosine_zero", cosine_zero::<S>),
        ("deletion_isolation", deletion_isolation::<S>),
        ("batch_validation", batch_validation::<S>),
    ];
    let mut failures = Vec::new();
    for (name, check) in checks {
        let dir = base.join(name);
        if let Err(msg) = check(&dir) {
            failures.push((name, msg));
        }
    }
    failures
}
