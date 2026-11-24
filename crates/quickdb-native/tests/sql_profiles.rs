use quickdb_api::*;
use quickdb_native::NativeStore;
use rusqlite::{params, Connection};
use std::hint::black_box;
use std::time::Instant;

fn directory() -> tempfile::TempDir {
    let root = std::env::var_os("QUICKDB_DATA").unwrap_or_else(|| std::env::temp_dir().into());
    tempfile::tempdir_in(root).unwrap()
}

fn word(id: u64) -> String {
    let head = if id.is_multiple_of(2) { "ab" } else { "كت" };
    let mut word = format!("{head}{:06x}", id / 2);
    if id.is_multiple_of(997) {
        let len = 2 + (id / 997) as usize % 11;
        word = word.chars().take(len).collect();
        while word.chars().count() < len {
            word.push('a');
        }
    }
    word
}

fn table(indexed: bool) -> TableDef {
    TableDef {
        name: "words".into(),
        columns: vec![ColumnDef {
            name: "word".into(),
            ty: ColumnType::Text,
            indexed,
        }],
    }
}

fn stats(label: &str, mut ns: Vec<u128>) {
    ns.sort_unstable();
    println!(
        "SQL_PROFILE {label} n={} median_us={:.3} p99_us={:.3} mean_us={:.3}",
        ns.len(),
        ns[ns.len() / 2] as f64 / 1000.0,
        ns[ns.len() * 99 / 100] as f64 / 1000.0,
        ns.iter().sum::<u128>() as f64 / ns.len() as f64 / 1000.0
    );
}

fn measure(mut f: impl FnMut(), n: usize) -> Vec<u128> {
    (0..n)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_nanos()
        })
        .collect()
}

fn plan(db: &Connection, label: &str, sql: &str, values: &[&dyn rusqlite::ToSql]) {
    let mut stmt = db.prepare(&format!("EXPLAIN QUERY PLAN {sql}")).unwrap();
    let rows = stmt.query_map(values, |r| r.get::<_, String>(3)).unwrap();
    for row in rows {
        println!("SQL_PLAN {label} {}", row.unwrap());
    }
}

#[test]
fn sql_profiles_140k_words_500k_edges() {
    let dir = directory();
    let mut s = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
    s.define_table(&table(true)).unwrap();
    for start in (0..140_000).step_by(1000) {
        let rows: Vec<_> = (start..start + 1000)
            .map(|id| Row {
                id,
                values: vec![Value::Text(word(id))],
            })
            .collect();
        s.upsert_rows("words", &rows).unwrap();
    }
    for start in (0..500_000).step_by(1000) {
        let edges: Vec<_> = (start..start + 1000)
            .map(|id| Edge {
                src: id / 5,
                rel: format!("r{}", id % 3),
                dst: 100_000 + id,
                weight: 1.0,
            })
            .collect();
        s.upsert_edges(&edges).unwrap();
    }
    let db = Connection::open(dir.path().join("quickdb.sqlite")).unwrap();
    // As NativeStore does on open; without it STAT4 re-prepares every cached statement.
    db.set_db_config(
        rusqlite::config::DbConfig::SQLITE_DBCONFIG_ENABLE_QPSG,
        true,
    )
    .unwrap();
    let old = "SELECT id,c_word FROM t_words WHERE substr(CAST(c_word AS BLOB),1,?1)=?2 ORDER BY id LIMIT ?3";
    let range = "SELECT id,c_word FROM t_words WHERE id IN (SELECT id FROM t_words WHERE c_word COLLATE BINARY >= CAST(?1 AS TEXT) AND c_word COLLATE BINARY < CAST(?2 AS TEXT) ORDER BY id LIMIT CAST(?3 AS INTEGER)) ORDER BY id";
    let p = "ab00000";
    plan(
        &db,
        "prefix_old",
        old,
        &[&(p.len() as i64), &p.as_bytes(), &100],
    );
    plan(
        &db,
        "prefix_range",
        range,
        &[&p.as_bytes(), &b"ab00001".as_slice(), &100],
    );
    let mut times = Vec::new();
    let mut legacy_times = Vec::new();
    let mut legacy_stmt = db.prepare_cached(old).unwrap();
    for i in 0..500 {
        let prefix = format!(
            "{}{:05x}",
            if i % 2 == 0 { "ab" } else { "كت" },
            (i * 131) % 4375
        );
        let start = Instant::now();
        let expected: Vec<Row> = legacy_stmt
            .query_map(params![prefix.len() as i64, prefix.as_bytes(), 100], |r| {
                Ok(Row {
                    id: r.get(0)?,
                    values: vec![Value::Text(r.get(1)?)],
                })
            })
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        legacy_times.push(start.elapsed().as_nanos());
        let f = Filter::TextPrefix {
            column: "word".into(),
            prefix,
        };
        let start = Instant::now();
        let rows = s.select("words", &f, 100).unwrap();
        times.push(start.elapsed().as_nanos());
        assert!(rows.len() <= 16);
        assert_eq!(rows, expected);
    }
    stats("prefix_legacy", legacy_times);
    stats("prefix_public", times);
    let eq = Filter::Eq {
        column: "word".into(),
        value: Value::Text(word(70_000)),
    };
    stats(
        "eq_public",
        measure(
            || {
                black_box(s.select("words", &eq, 100).unwrap());
            },
            2000,
        ),
    );
    stats(
        "get_row_public",
        measure(
            || {
                black_box(s.get_row("words", 70_000).unwrap());
            },
            2000,
        ),
    );
    stats(
        "neighbors_public_out",
        measure(
            || {
                black_box(s.neighbors(50_000, None, Direction::Out, 100).unwrap());
            },
            2000,
        ),
    );
    stats(
        "neighbors_public_in",
        measure(
            || {
                black_box(s.neighbors(350_000, None, Direction::In, 100).unwrap());
            },
            2000,
        ),
    );
    stats(
        "neighbors_public_both",
        measure(
            || {
                black_box(s.neighbors(50_000, None, Direction::Both, 100).unwrap());
            },
            2000,
        ),
    );
    let eq_sql = "SELECT id,c_word FROM t_words WHERE c_word=?1 ORDER BY id LIMIT ?2";
    let edge_sql =
        "SELECT src,rel,dst,weight FROM edges WHERE src=?1 ORDER BY rel,src,dst LIMIT ?2";
    plan(&db, "eq", eq_sql, &[&word(70_000), &100]);
    plan(&db, "neighbors_out", edge_sql, &[&50_000, &100]);
    profile_eq(&db, eq_sql, "before");
    profile_edges(&db, edge_sql, "before");
    let eq_after =
        "SELECT id,c_word FROM t_words WHERE c_word=?1 ORDER BY id LIMIT CAST(?2 AS INTEGER)";
    let edge_after = "SELECT src,rel,dst,weight FROM edges WHERE src=?1 ORDER BY rel,src,dst LIMIT CAST(?2 AS INTEGER)";
    plan(&db, "eq_after", eq_after, &[&word(70_000), &100]);
    plan(&db, "neighbors_out_after", edge_after, &[&50_000, &100]);
    profile_eq(&db, eq_after, "after");
    profile_edges(&db, edge_after, "after");
    for (label, sql, text) in [
        ("eq_limit_parameter", eq_sql, true),
        ("eq_limit_arithmetic", "SELECT id,c_word FROM t_words WHERE c_word=?1 ORDER BY id LIMIT (?2+0)", true),
        ("eq_limit_cast", "SELECT id,c_word FROM t_words WHERE c_word=?1 ORDER BY id LIMIT CAST(?2 AS INTEGER)", true),
        ("neighbors_limit_parameter", edge_sql, false),
        ("neighbors_limit_arithmetic", "SELECT src,rel,dst,weight FROM edges WHERE src=?1 ORDER BY rel,src,dst LIMIT (?2+0)", false),
        ("neighbors_limit_cast", "SELECT src,rel,dst,weight FROM edges WHERE src=?1 ORDER BY rel,src,dst LIMIT CAST(?2 AS INTEGER)", false),
    ] {
        let mut stmt = db.prepare_cached(sql).unwrap();
        stats(label, measure(|| {
            let key = if text { rusqlite::types::Value::Text(word(70_000)) } else { 50_000.into() };
            let mut rows = stmt.query(params![key,100]).unwrap();
            while let Some(r) = rows.next().unwrap() { black_box(r.get::<_,u64>(0).unwrap()); }
        }, 2000));
        println!("SQL_COUNTER {label} runs={} reprepares={} sorts={} vm_steps={}", stmt.get_status(rusqlite::StatementStatus::Run), stmt.get_status(rusqlite::StatementStatus::RePrepare), stmt.get_status(rusqlite::StatementStatus::Sort), stmt.get_status(rusqlite::StatementStatus::VmStep));
        if label.ends_with("_cast") {
            assert_eq!(stmt.get_status(rusqlite::StatementStatus::RePrepare), 0);
            assert_eq!(stmt.get_status(rusqlite::StatementStatus::Sort), 0);
        }
    }
}

fn profile_eq(db: &Connection, sql: &str, phase: &str) {
    let key = word(70_000);
    let covering_sql = sql.replace("id,c_word", "id");
    stats(
        &format!("eq_{phase}_prepare_uncached"),
        measure(
            || {
                black_box(db.prepare(sql).unwrap());
            },
            2000,
        ),
    );
    stats(
        &format!("eq_{phase}_cache_lookup"),
        measure(
            || {
                black_box(db.prepare_cached(sql).unwrap());
            },
            2000,
        ),
    );
    stats(
        &format!("eq_{phase}_covering_probe_order_limit"),
        measure(
            || {
                let mut stmt = db.prepare_cached(&covering_sql).unwrap();
                let ids: Vec<i64> = stmt
                    .query_map(params![key, 100], |r| r.get(0))
                    .unwrap()
                    .map(|r| r.unwrap())
                    .collect();
                black_box(ids);
            },
            2000,
        ),
    );
    stats(
        &format!("eq_{phase}_fetch_without_values"),
        measure(
            || {
                let mut stmt = db.prepare_cached(sql).unwrap();
                let mut rows = stmt.query(params![key, 100]).unwrap();
                while let Some(r) = rows.next().unwrap() {
                    black_box(r.get::<_, i64>(0).unwrap());
                }
            },
            2000,
        ),
    );
    stats(
        &format!("eq_{phase}_fetch_build_rows"),
        measure(
            || {
                let mut stmt = db.prepare_cached(sql).unwrap();
                let rows: Vec<_> = stmt
                    .query_map(params![key, 100], |r| {
                        Ok(Row {
                            id: r.get::<_, u64>(0)?,
                            values: vec![Value::Text(r.get(1)?)],
                        })
                    })
                    .unwrap()
                    .map(|r| r.unwrap())
                    .collect();
                black_box(rows);
            },
            2000,
        ),
    );
    stats(
        &format!("eq_{phase}_without_order_limit"),
        measure(
            || {
                let mut stmt = db
                    .prepare_cached("SELECT id,c_word FROM t_words WHERE c_word=?1")
                    .unwrap();
                let rows: Vec<_> = stmt
                    .query_map([&key], |r| {
                        Ok((r.get::<_, u64>(0)?, r.get::<_, String>(1)?))
                    })
                    .unwrap()
                    .map(|r| r.unwrap())
                    .collect();
                black_box(rows);
            },
            2000,
        ),
    );
    stats(
        &format!("eq_{phase}_rowid_fetch_build"),
        measure(
            || {
                let mut stmt = db
                    .prepare_cached("SELECT id,c_word FROM t_words WHERE id=?1")
                    .unwrap();
                black_box(
                    stmt.query_row([70_000], |r| {
                        Ok(Row {
                            id: r.get::<_, u64>(0)?,
                            values: vec![Value::Text(r.get(1)?)],
                        })
                    })
                    .unwrap(),
                );
            },
            2000,
        ),
    );
}

fn profile_edges(db: &Connection, sql: &str, phase: &str) {
    let covering_sql = sql.replace("src,rel,dst,weight", "src,rel,dst");
    stats(
        &format!("neighbors_{phase}_prepare_uncached"),
        measure(
            || {
                black_box(db.prepare(sql).unwrap());
            },
            2000,
        ),
    );
    stats(
        &format!("neighbors_{phase}_cache_lookup"),
        measure(
            || {
                black_box(db.prepare_cached(sql).unwrap());
            },
            2000,
        ),
    );
    stats(
        &format!("neighbors_{phase}_covering_probe_order_limit"),
        measure(
            || {
                let mut stmt = db.prepare_cached(&covering_sql).unwrap();
                let mut rows = stmt.query([50_000, 100]).unwrap();
                while let Some(r) = rows.next().unwrap() {
                    black_box(r.get::<_, u64>(2).unwrap());
                }
            },
            2000,
        ),
    );
    stats(
        &format!("neighbors_{phase}_fetch_without_values"),
        measure(
            || {
                let mut stmt = db.prepare_cached(sql).unwrap();
                let mut rows = stmt.query([50_000, 100]).unwrap();
                while let Some(r) = rows.next().unwrap() {
                    black_box(r.get::<_, u64>(0).unwrap());
                }
            },
            2000,
        ),
    );
    stats(
        &format!("neighbors_{phase}_fetch_build_edges"),
        measure(
            || {
                let mut stmt = db.prepare_cached(sql).unwrap();
                let edges: Vec<_> = stmt
                    .query_map([50_000, 100], |r| {
                        Ok(Edge {
                            src: r.get(0)?,
                            rel: r.get(1)?,
                            dst: r.get(2)?,
                            weight: r.get(3)?,
                        })
                    })
                    .unwrap()
                    .map(|r| r.unwrap())
                    .collect();
                black_box(edges);
            },
            2000,
        ),
    );
    stats(
        &format!("neighbors_{phase}_without_order_limit"),
        measure(
            || {
                let mut stmt = db
                    .prepare_cached("SELECT src,rel,dst,weight FROM edges WHERE src=?1")
                    .unwrap();
                let edges: Vec<_> = stmt
                    .query_map([50_000], |r| {
                        Ok(Edge {
                            src: r.get(0)?,
                            rel: r.get(1)?,
                            dst: r.get(2)?,
                            weight: r.get(3)?,
                        })
                    })
                    .unwrap()
                    .map(|r| r.unwrap())
                    .collect();
                black_box(edges);
            },
            2000,
        ),
    );
}
