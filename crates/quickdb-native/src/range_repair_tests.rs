#[test]
fn corpus_range_query_plan_and_late_materialization() {
    use std::time::Instant;
    let dir = directory();
    let mut store = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
    // Written out instead of calling `quickdb_bench::benchmark_table()`: the packaged crate has no
    // `quickdb-bench` dev-dependency. Keep in step with that function.
    let col = |name: &str, ty: ColumnType, indexed: bool| ColumnDef {
        name: name.into(),
        ty,
        indexed,
    };
    let def = TableDef {
        name: "docs".into(),
        columns: vec![
            col("kind", ColumnType::Text, true),
            col("title", ColumnType::Text, false),
            col("size", ColumnType::Int, true),
            col("score", ColumnType::Float, false),
            col("body", ColumnType::Text, false),
        ],
    };
    store.define_table(&def).unwrap();
    // Corpus labels only; the ordinal in column 5 of the fixture picks one.
    let kinds = [
        "corpus-0",
        "corpus-1",
        "corpus-2",
        "corpus-3",
        "corpus-4",
        "corpus-5",
        "corpus-6",
        "corpus-7",
        "corpus-8",
        "corpus-9",
    ];
    let rows: Vec<_> = include_str!("../tests/fixtures/corpus-shapes.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .enumerate()
        .map(|(index, line)| {
            let v: Vec<usize> = line.split('\t').map(|v| v.parse().unwrap()).collect();
            Row {
                id: 1_000_000 + index as u64,
                values: vec![
                    Value::Text(kinds[v[4]].into()),
                    Value::Text("x".repeat(v[2])),
                    Value::Int(v[0] as i64),
                    Value::Float(v[3] as f64),
                    Value::Text("x".repeat(v[1])),
                ],
            }
        })
        .collect();
    assert_eq!(rows.len(), 22666);
    for batch in rows.chunks(1000) {
        store.upsert_rows("docs", batch).unwrap();
    }
    let old_sql = format!(
        "SELECT {} FROM t_docs WHERE c_size BETWEEN ?1 AND ?2 ORDER BY id LIMIT ?3",
        projection(&def)
    );
    let late_sql = format!("SELECT {} FROM t_docs WHERE id IN (SELECT id FROM t_docs WHERE c_size BETWEEN ?1 AND ?2 ORDER BY id LIMIT ?3) ORDER BY id", projection(&def));
    for (label, sql) in [("baseline", &old_sql), ("late", &late_sql)] {
        let mut stmt = store
            .db
            .prepare(&format!("EXPLAIN QUERY PLAN {sql}"))
            .unwrap();
        let plan: Vec<String> = stmt
            .query_map(params![1100i64, 1200i64, 100i64], |r| r.get(3))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        println!("CORPUS_RANGE_PLAN {label} {}", plan.join(" | "));
        assert!(plan.iter().any(|line| line.contains("idx_4_docs_size")));
        if label == "late" {
            assert!(plan.iter().any(|line| line.contains("COVERING INDEX")));
        }
    }
    let mut rng = Lcg(0x5142494453);
    for _ in 0..10000 {
        rng.next_u32();
    }
    let mut old_times = Vec::new();
    let mut late_times = Vec::new();
    let mut actual_times = Vec::new();
    let mut matches = Vec::new();
    let mut old = store.db.prepare_cached(&old_sql).unwrap();
    let mut late = store.db.prepare_cached(&late_sql).unwrap();
    for _ in 0..500 {
        let row = &rows[rng.next_u32() as usize % rows.len()];
        let Value::Int(min) = row.values[2] else {
            unreachable!()
        };
        let max = min + 100;
        let count: usize = store
            .db
            .query_row(
                "SELECT count(*) FROM t_docs WHERE c_size BETWEEN ? AND ?",
                params![min, max],
                |r| r.get(0),
            )
            .unwrap();
        matches.push(count);
        let start = Instant::now();
        let expected: Vec<_> = old
            .query_map(params![min, max, 100i64], |r| read_row(r, &def))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        old_times.push(start.elapsed().as_secs_f64() * 1e6);
        let start = Instant::now();
        let optimized: Vec<_> = late
            .query_map(params![min, max, 100i64], |r| read_row(r, &def))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        late_times.push(start.elapsed().as_secs_f64() * 1e6);
        assert_eq!(expected, optimized);
        let start = Instant::now();
        let actual = store
            .select(
                "docs",
                &Filter::IntRange {
                    column: "size".into(),
                    min,
                    max,
                },
                100,
            )
            .unwrap();
        actual_times.push(start.elapsed().as_secs_f64() * 1e6);
        assert_eq!(actual, expected);
    }
    matches.sort_unstable();
    old_times.sort_by(f64::total_cmp);
    late_times.sort_by(f64::total_cmp);
    actual_times.sort_by(f64::total_cmp);
    println!("CORPUS_RANGE rows={} samples=500 matching_p50={} matching_max={} baseline_p50_us={:.3} late_p50_us={:.3} actual_select_p50_us={:.3} limit=100 payload_shapes=fixture text=placeholder", rows.len(), matches[249], matches[499], old_times[249], late_times[249], actual_times[249]);
}
