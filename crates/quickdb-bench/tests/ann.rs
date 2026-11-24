use quickdb_api::{Error, Metric};
use quickdb_bench::{ann, exact_top10_metric, run, MemStore};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
fn fixture() -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "ann-fixture-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&path).expect("fixture directory");
    let base: Vec<Vec<f32>> = (0..12).map(|i| vec![i as f32, 1.0]).collect();
    let queries = vec![vec![-1.0, 1.0], vec![12.0, 1.0]];
    write_fvecs(&path.join("base.fvecs"), &base);
    write_fvecs(&path.join("query.fvecs"), &queries);
    let mut truth = std::fs::File::create(path.join("truth.ivecs")).expect("truth");
    for ids in [(0i32..10).collect::<Vec<_>>(), (2i32..12).rev().collect()] {
        truth.write_all(&10i32.to_le_bytes()).expect("dimension");
        for id in ids {
            truth.write_all(&id.to_le_bytes()).expect("id");
        }
    }
    path
}
fn write_fvecs(path: &Path, vectors: &[Vec<f32>]) {
    let mut file = std::fs::File::create(path).expect("fvecs");
    for vector in vectors {
        file.write_all(&(vector.len() as i32).to_le_bytes())
            .expect("dimension");
        for value in vector {
            file.write_all(&value.to_le_bytes()).expect("value");
        }
    }
}
#[test]
fn generated_full_and_prefix_run_against_memstore() {
    let path = fixture();
    for (limit, source) in [(12, "ivecs"), (7, "exact_prefix")] {
        let data = ann(
            &path.join("base.fvecs"),
            &path.join("query.fvecs"),
            &path.join("truth.ivecs"),
            limit,
            2,
            Metric::L2,
        )
        .expect("ann");
        assert_eq!(data.description["truth_source"], source);
        assert_eq!(data.vectors[0].0, 0);
        assert_eq!(data.rows[0].id, 0);
        assert!(data
            .edges
            .iter()
            .all(|e| e.src < limit as u64 && e.dst < limit as u64));
        if limit == 7 {
            assert!(data.truth.is_none());
            assert_eq!(
                exact_top10_metric(&data.vectors, &[12.0, 1.0], None, Metric::L2),
                vec![6, 5, 4, 3, 2, 1, 0]
            );
        } else {
            assert_eq!(
                data.truth.as_ref().expect("file truth")[1],
                vec![11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
            );
        }
        let report = run::<MemStore>(&data, &path.join(format!("db-{limit}"))).expect("benchmark");
        assert_eq!(report.metric, "l2");
        assert_eq!(report.rows, limit as u64);
        for ef in [32, 64, 128, 256] {
            assert_eq!(
                report.queries[&format!("knn_ef{ef}")].recall_at_10,
                Some(1.0)
            );
            assert_eq!(report.queries[&format!("knn_ef{ef}")].samples, 2);
        }
        assert_eq!(report.queries["knn_filtered_ef64"].recall_at_10, Some(1.0));
    }
    std::fs::remove_dir_all(path).expect("cleanup");
}
#[test]
fn malformed_files_and_counts_are_schema_errors() {
    let path = fixture();
    let load = |n, q| {
        ann(
            &path.join("base.fvecs"),
            &path.join("query.fvecs"),
            &path.join("truth.ivecs"),
            n,
            q,
            Metric::L2,
        )
    };
    for (n, q) in [(0, 2), (13, 2), (12, 0), (12, 3)] {
        assert!(matches!(load(n, q), Err(Error::Schema(_))));
    }
    write_fvecs(&path.join("query.fvecs"), &[vec![0.0, f32::NAN]]);
    assert!(matches!(load(12, 1), Err(Error::Schema(_))));
    write_fvecs(&path.join("query.fvecs"), &[vec![0.0]]);
    assert!(matches!(load(12, 1), Err(Error::Schema(_))));
    write_fvecs(&path.join("query.fvecs"), &[vec![0.0, 1.0], vec![1.0]]);
    assert!(matches!(load(12, 1), Err(Error::Schema(_))));
    write_fvecs(&path.join("query.fvecs"), &[vec![0.0, 1.0]]);
    std::fs::write(path.join("truth.ivecs"), [0u8; 4]).expect("corrupt");
    assert!(matches!(load(12, 1), Err(Error::Schema(_))));
    std::fs::write(path.join("base.fvecs"), [2u8, 0, 0, 0, 0]).expect("truncate");
    assert!(matches!(load(12, 1), Err(Error::Schema(_))));
    std::fs::remove_dir_all(path).expect("cleanup");
}
#[test]
fn l2_truth_does_not_accidentally_rank_by_cosine() {
    let data = vec![(0, vec![100.0, 0.0]), (1, vec![1.0, 1.0])];
    assert_eq!(
        exact_top10_metric(&data, &[1.0, 0.0], None, Metric::L2),
        vec![1, 0]
    );
    assert_eq!(
        exact_top10_metric(&data, &[1.0, 0.0], None, Metric::Cosine),
        vec![0, 1]
    );
}
