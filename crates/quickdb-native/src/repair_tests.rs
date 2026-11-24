#[cfg(test)]
mod tests {
    use super::*;
    use quickdb_api::conformance::Lcg;

    #[test]
    fn levels_follow_geometric_distribution_100k() {
        let mut h = Hnsw::new(1, Metric::L2, 16, 32);
        for id in 0..100_000 {
            h.insert(id, &[id as f32]);
        }
        for level in 0..=5 {
            let count = h.links.iter().filter(|layers| layers.len() > level).count();
            let expected = 100_000.0 / 16f64.powi(level as i32);
            println!("LEVEL_GEOMETRIC nodes=100000 m=16 level_ge={level} actual={count} expected={expected:.6}");
            assert!((count as f64 - expected).abs() <= 6.0 * expected.sqrt() + 1.0);
        }
    }

    #[test]
    fn entry_is_top_and_reaches_every_node_20k() {
        let mut rng = Lcg(9927);
        let data: Vec<_> = (0..20_000).map(|id| (id, rng.vector(32))).collect();
        for bulk in [false, true] {
            let mut h = Hnsw::new(32, Metric::Cosine, 16, 200);
            if bulk {
                h.insert_many(
                    &data
                        .iter()
                        .map(|(id, v)| (*id, v.as_slice()))
                        .collect::<Vec<_>>(),
                );
            } else {
                for (id, vector) in &data {
                    h.insert(*id, vector);
                }
            }
            let entry = h.entry.unwrap();
            assert_eq!(h.links[entry].len(), h.top + 1);
            assert!(h.links.iter().all(|layers| layers.len() <= h.top + 1));
            let mut seen = vec![false; h.ids.len()];
            let mut stack = vec![entry];
            seen[entry] = true;
            while let Some(node) = stack.pop() {
                for &next in &h.links[node][0] {
                    if !seen[next] {
                        seen[next] = true;
                        stack.push(next);
                    }
                }
            }
            let reachable = seen.iter().filter(|seen| **seen).count();
            println!("ENTRY_REACHABILITY bulk={bulk} nodes=20000 entry={entry} top={} reached={reachable}", h.top);
            assert_eq!(reachable, h.ids.len());
        }
    }

    #[test]
    fn parallel_build_is_schedule_independent() {
        let mut rng = Lcg(1234);
        let data: Vec<_> = (0..5000).map(|id| (id, rng.vector(32))).collect();
        let mut items: Vec<_> = data.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        items.extend(items[..100].to_vec());
        let mut one = Hnsw::new(32, Metric::Cosine, 16, 200);
        let mut four = Hnsw::new(32, Metric::Cosine, 16, 200);
        one.insert_many_with_workers(&items, 1);
        four.insert_many_with_workers(&items, 4);
        assert_eq!(one.data, four.data);
        assert_eq!(one.links, four.links);
        assert_eq!(one.active, four.active);
        assert_eq!(four.len(), 5000);
        assert!(four.valid(32, Metric::Cosine, 16, 200));
        for _ in 0..20 {
            let q = rng.vector(32);
            assert_eq!(one.search(&q, 10, 128), four.search(&q, 10, 128));
        }
        println!("PARALLEL_DETERMINISM workers=1,4 nodes=5100 active=5000 graphs_identical=true");
    }

    #[test]
    fn rejects_excess_layers_on_a_non_entry_node() {
        let mut h = Hnsw::new(1, Metric::L2, 16, 32);
        h.insert(0, &[0.]);
        h.insert(1, &[1.]);
        assert!(h.valid(1, Metric::L2, 16, 32));
        let node = usize::from(h.entry == Some(0));
        h.links[node].resize(h.top + 2, Vec::new());
        assert!(!h.valid(1, Metric::L2, 16, 32));
        println!("SNAPSHOT_LAYER_VALIDATION non_entry_excess_layers_rejected=true");
    }

    #[test]
    fn compaction_threshold_is_strictly_twenty_percent() {
        let mut h = Hnsw::new(1, Metric::L2, 16, 32);
        for id in 0..100 {
            h.insert(id, &[id as f32]);
        }
        for id in 0..25 {
            h.insert(id, &[id as f32 + 1.]);
        }
        assert_eq!(h.node_count(), 125);
        assert!(!h.needs_compaction());
        h.insert(25, &[26.]);
        assert!(h.needs_compaction());
    }

    #[test]
    fn checksummed_snapshot_with_invalid_non_entry_layers_rebuilds() {
        use quickdb_api::{OpenOptions, Store, VectorIndexDef};
        let root = std::env::var_os("QUICKDB_DATA")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let dir = tempfile::tempdir_in(root).unwrap();
        let mut store = crate::NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
        store
            .define_vector_index(&VectorIndexDef {
                name: "emb".into(),
                dim: 1,
                metric: Metric::L2,
                m: 16,
                ef_construction: 32,
            })
            .unwrap();
        store
            .upsert_vectors("emb", &[(0, &[0.]), (1, &[1.])])
            .unwrap();
        let h = store.indexes.get_mut("emb").unwrap();
        let node = usize::from(h.entry == Some(0));
        h.links[node].resize(h.top + 2, Vec::new());
        store.flush().unwrap();
        drop(store);
        // The checksum is valid, so the decoder's structural validation is what rejects it.
        assert!(crate::load_snapshot(&dir.path().join("emb.hnsw")).is_none());
        let store = crate::NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
        assert_eq!(store.index_rebuilds(), 1);
        assert_eq!(store.knn("emb", &[1.], 1, 32).unwrap()[0].id, 1);
        println!("SNAPSHOT_LAYER_LOAD checksum_valid=true invalid_non_entry_layers=true rebuilt=1");
    }
}
