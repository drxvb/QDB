//! Filtered vector search: only allowed ids, `min(k, allowed)` results, deletions,
//! replacements, and snapshot reuse across reopen.
use quickdb_api::{
    ColumnDef, ColumnType, Filter, Metric, OpenOptions, Row, Store, TableDef, Value, VectorIndexDef,
};
use quickdb_native::NativeStore;
use std::collections::HashSet;

const N: u64 = 6000;
const DIM: usize = 16;

fn vector(id: u64) -> Vec<f32> {
    let mut h = id.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x5151;
    (0..DIM)
        .map(|_| {
            h ^= h << 13;
            h ^= h >> 7;
            h ^= h << 17;
            (h >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

/// Column `p` holds "y" for ids with `id % 1000 < per_mille`.
const PER_MILLE: [u64; 5] = [1, 10, 100, 500, 900];

fn build(dir: &std::path::Path) -> NativeStore {
    let mut s = NativeStore::open(dir, &OpenOptions::default()).unwrap();
    s.define_table(&TableDef {
        name: "docs".into(),
        columns: (0..PER_MILLE.len())
            .map(|i| ColumnDef {
                name: format!("p{i}"),
                ty: ColumnType::Text,
                indexed: true,
            })
            .collect(),
    })
    .unwrap();
    s.define_vector_index(&VectorIndexDef {
        name: "emb".into(),
        dim: DIM,
        metric: Metric::L2,
        m: 16,
        ef_construction: 100,
    })
    .unwrap();
    let rows: Vec<Row> = (0..N)
        .map(|id| Row {
            id,
            values: PER_MILLE
                .iter()
                .map(|&p| Value::Text(if id % 1000 < p { "y" } else { "n" }.into()))
                .collect(),
        })
        .collect();
    s.upsert_rows("docs", &rows).unwrap();
    let vs: Vec<_> = (0..N).map(vector).collect();
    let items: Vec<_> = vs
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v.as_slice()))
        .collect();
    s.upsert_vectors("emb", &items).unwrap();
    s
}

fn allowed(p: usize) -> HashSet<u64> {
    (0..N).filter(|id| id % 1000 < PER_MILLE[p]).collect()
}

fn filter(p: usize) -> Filter {
    Filter::Eq {
        column: format!("p{p}"),
        value: Value::Text("y".into()),
    }
}

fn check_all(s: &NativeStore, live: &dyn Fn(u64) -> Option<Vec<f32>>) {
    for p in 0..PER_MILLE.len() {
        let ok: HashSet<u64> = allowed(p)
            .into_iter()
            .filter(|&id| live(id).is_some())
            .collect();
        for qi in 0..10u64 {
            let q = vector(1_000_000 + qi);
            for (k, ef) in [(10, 64), (50, 400), (ok.len() + 5, N as usize * 2)] {
                let hits = s
                    .knn_filtered("emb", &q, k, ef, "docs", &filter(p))
                    .unwrap();
                assert!(
                    hits.iter().all(|h| ok.contains(&h.id)),
                    "p={p}: hit outside allowed set"
                );
                assert_eq!(hits.len(), k.min(ok.len()), "p={p} k={k} ef={ef}");
                for h in &hits {
                    let v = live(h.id).unwrap();
                    let d: f32 = v.iter().zip(&q).map(|(a, b)| (a - b) * (a - b)).sum();
                    assert!(
                        (d - h.distance).abs() <= 1e-4 * d.max(1.0),
                        "stale distance for {}",
                        h.id
                    );
                }
                assert!(hits.windows(2).all(|w| w[0].distance <= w[1].distance));
            }
        }
    }
}

#[test]
fn filtered_search_respects_set_count_and_replacements() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = build(dir.path());
    check_all(&s, &|id| (id < N).then(|| vector(id)));
    // Replace a third of the vectors; the old positions become tombstones.
    let replaced: Vec<_> = (0..N)
        .filter(|id| id.is_multiple_of(3))
        .map(|id| (id, vector(id + 7_777_777)))
        .collect();
    let items: Vec<_> = replaced.iter().map(|(id, v)| (*id, v.as_slice())).collect();
    s.upsert_vectors("emb", &items).unwrap();
    let now = |id: u64| {
        (id < N).then(|| {
            if id.is_multiple_of(3) {
                vector(id + 7_777_777)
            } else {
                vector(id)
            }
        })
    };
    check_all(&s, &now);
    // Rows deleted from the table must leave the filter.
    let gone: Vec<u64> = (0..N)
        .filter(|id| id % 1000 < 900 && id.is_multiple_of(7))
        .collect();
    s.delete_rows("docs", &gone).unwrap();
    for p in 0..PER_MILLE.len() {
        let ok: HashSet<u64> = allowed(p)
            .into_iter()
            .filter(|id| !id.is_multiple_of(7))
            .collect();
        let hits = s
            .knn_filtered("emb", &vector(42), 20, 64, "docs", &filter(p))
            .unwrap();
        assert!(hits.iter().all(|h| ok.contains(&h.id)));
        assert_eq!(hits.len(), 20.min(ok.len()));
    }
}

#[test]
fn filtered_recall_is_high_on_graph_path() {
    let dir = tempfile::tempdir().unwrap();
    let s = build(dir.path());
    for p in 2..PER_MILLE.len() {
        let ok = allowed(p);
        let mut total = 0.0;
        for qi in 0..50u64 {
            let q = vector(2_000_000 + qi);
            let mut truth: Vec<(f32, u64)> = ok
                .iter()
                .map(|&id| {
                    (
                        vector(id)
                            .iter()
                            .zip(&q)
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum(),
                        id,
                    )
                })
                .collect();
            truth.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let t: HashSet<u64> = truth.iter().take(10).map(|x| x.1).collect();
            let hits = s
                .knn_filtered("emb", &q, 10, 64, "docs", &filter(p))
                .unwrap();
            total += hits.iter().filter(|h| t.contains(&h.id)).count() as f64 / 10.0;
        }
        assert!(total / 50.0 >= 0.95, "p={p} recall {}", total / 50.0);
    }
}

#[test]
fn snapshot_loads_and_filtered_search_works_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = build(dir.path());
    s.flush().unwrap();
    let before: Vec<_> = (0..PER_MILLE.len())
        .map(|p| {
            s.knn_filtered("emb", &vector(9), 10, 64, "docs", &filter(p))
                .unwrap()
        })
        .collect();
    drop(s);
    let s = NativeStore::open(
        dir.path(),
        &OpenOptions {
            create: false,
            sync_each_write: false,
        },
    )
    .unwrap();
    assert_eq!(s.index_rebuilds(), 0, "snapshot was not reused");
    for (p, b) in before.iter().enumerate() {
        assert_eq!(
            &s.knn_filtered("emb", &vector(9), 10, 64, "docs", &filter(p))
                .unwrap(),
            b
        );
    }
}

#[test]
fn empty_and_zero_cases() {
    let dir = tempfile::tempdir().unwrap();
    let s = build(dir.path());
    let none = Filter::Eq {
        column: "p0".into(),
        value: Value::Text("nothing".into()),
    };
    assert!(s
        .knn_filtered("emb", &vector(1), 10, 64, "docs", &none)
        .unwrap()
        .is_empty());
    assert!(s
        .knn_filtered("emb", &vector(1), 0, 64, "docs", &filter(4))
        .unwrap()
        .is_empty());
    assert!(
        s.knn_filtered("emb", &vector(1), 10, 64, "docs", &Filter::All)
            .unwrap()
            .len()
            == 10
    );
}
