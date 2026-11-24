//! Regenerate tests/bound_expected.rs using the frozen pre-membound collection rule.
//! Run remotely: cargo run -p quickdb-native --release --example bound_reference.
#[cfg(not(test))]
#[path = "../tests/bound_filtered.rs"]
mod fixture;
#[cfg(not(test))]
#[allow(dead_code)]
#[path = "../src/hnsw.rs"]
mod hnsw;

#[cfg(not(test))]
mod reference {
    use super::{fixture, hnsw};
    use quickdb_api::*;
    use rusqlite::{params_from_iter, Connection};
    use std::io::Write;

    // Frozen pre-membound algorithm, specialized to the fixture's Eq(Int) filters.
    fn old_filtered(
        h: &hnsw::Hnsw,
        db: &Connection,
        query: &[f32],
        k: usize,
        ef_search: usize,
        column: &str,
    ) -> (Vec<Hit>, &'static str) {
        const PROBE_IDS: usize = 128;
        let ef = ef_search.max(k);
        let cap = h.capacity();
        {
            let slots = vec!["?"; PROBE_IDS].join(",");
            let mut probe = db.prepare_cached(&format!("SELECT id FROM t_docs NOT INDEXED WHERE id IN ({slots}) AND (\"c_{column}\"=?)")).expect("probe");
            probe
                .raw_bind_parameter(PROBE_IDS + 1, 1)
                .expect("filter bind");
            let mut matches = |nodes: &[usize], out: &mut [bool]| {
                for (chunk, flags) in nodes.chunks(PROBE_IDS).zip(out.chunks_mut(PROBE_IDS)) {
                    for slot in 0..PROBE_IDS {
                        probe
                            .raw_bind_parameter(
                                slot + 1,
                                chunk.get(slot).map_or(-1, |&n| h.id_at(n) as i64),
                            )
                            .expect("id bind");
                    }
                    let mut rows = probe.raw_query();
                    while let Some(row) = rows.next().expect("probe row") {
                        let id: u64 = row.get(0).expect("id");
                        if let Some(i) = chunk.iter().position(|&n| h.id_at(n) == id) {
                            flags[i] = true;
                        }
                    }
                }
            };
            let samples = 256.min(cap);
            let spread: Vec<usize> = (0..samples).map(|i| i * cap / samples).collect();
            let mut seen = vec![false; samples];
            matches(&spread, &mut seen);
            let estimate = seen.iter().filter(|&&b| b).count() as f64 / samples as f64;
            let lazy_from = (15.0 * ef as f64 / h.len() as f64).sqrt().max(0.02);
            if estimate >= lazy_from {
                let share: u128 = if estimate < 0.1 { 40 } else { 60 };
                let mut e = ((ef as u128 * share / 100) as usize).max(k);
                loop {
                    let (hits, _) = h.search_filtered(query, k, e, usize::MAX, 8, &mut matches);
                    if hits.len() >= k || e >= cap {
                        if hits.len() >= k {
                            return (hits, "lazy");
                        }
                        break;
                    }
                    e = e.saturating_mul(4).min(cap);
                }
            }
        }
        let mut stmt = db
            .prepare_cached(&format!("SELECT id FROM t_docs WHERE \"c_{column}\"=?"))
            .expect("ids");
        let mut rows = stmt.query(params_from_iter([1])).expect("query");
        let mut nodes = Vec::new();
        while let Some(row) = rows.next().expect("row") {
            if let Some(n) = h.node_of(row.get::<_, u64>(0).expect("id")) {
                nodes.push(n);
            }
        }
        let exact_up_to = (2.0 * (ef as f64 * h.len() as f64).sqrt()).max(1024.0) as usize;
        if nodes.len() <= exact_up_to {
            return (h.exact_nodes(query, k, &nodes), "exact");
        }
        let mut mask = vec![0u64; cap.div_ceil(64)];
        for &n in &nodes {
            mask[n / 64] |= 1 << (n % 64);
        }
        let mut e = ef;
        loop {
            let (hits, _) = h.search_filtered(query, k, e, usize::MAX, 1, |nodes, out| {
                for (&n, flag) in nodes.iter().zip(out) {
                    *flag = (mask[n / 64] >> (n % 64)) & 1 == 1;
                }
            });
            if hits.len() >= k.min(nodes.len()) {
                return (hits, "masked");
            }
            if e >= cap {
                return (h.exact_nodes(query, k, &nodes), "fallback");
            }
            e = e.saturating_mul(4).min(cap);
        }
    }

    pub fn generate() {
        let dir = tempfile::tempdir_in("/data").expect("fixture directory");
        let mut fixture = fixture::build(dir.path());
        let mut h = hnsw::Hnsw::new(16, Metric::L2, 8, 40);
        for vectors in [&fixture.vectors, &fixture.replacements] {
            let items: Vec<_> = vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
            h.insert_many(&items);
        }
        assert_eq!(h.capacity(), fixture::N + fixture.replacements.len());
        let db = Connection::open(dir.path().join("quickdb.sqlite")).expect("oracle DB");
        let mut output = std::io::BufWriter::new(
            std::fs::File::create("/data/bound_expected.rs").expect("fixture output"),
        );
        writeln!(output, "// Generated by bound_reference, frozen pre-membound collection.\n#[rustfmt::skip]\npub const EXPECTED: &[&[(u64, u32)]] = &[").expect("header");
        let mut counts = [0; 3];
        for qi in 0..200 {
            let query = fixture.rng.vector(16);
            let (k, ef) = [(1, 8), (7, 32), (17, 64), (31, 128)][qi % 4];
            for (path, column) in fixture::COLUMNS.iter().enumerate() {
                let (expected, actual_path) = old_filtered(&h, &db, &query, k, ef, column);
                assert_eq!(actual_path, *column, "oracle path");
                let actual = fixture
                    .store
                    .knn_filtered(
                        "emb",
                        &query,
                        k,
                        ef,
                        "docs",
                        &Filter::Eq {
                            column: (*column).into(),
                            value: Value::Int(1),
                        },
                    )
                    .expect("KNN");
                assert_eq!(actual, expected, "query={qi} path={column}");
                let bits: Vec<_> = expected
                    .iter()
                    .map(|hit| (hit.id, hit.distance.to_bits()))
                    .collect();
                writeln!(output, "    &{bits:?},").expect("fixture row");
                counts[path] += 1;
            }
        }
        writeln!(output, "];").expect("end");
        println!("BOUND oracle queries=600 exact={} masked={} lazy={} identical_ids_and_distances=true output=/data/bound_expected.rs", counts[0], counts[1], counts[2]);
    }
}

fn main() {
    #[cfg(not(test))]
    reference::generate();
}
