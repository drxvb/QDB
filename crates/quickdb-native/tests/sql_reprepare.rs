//! Which statement forms make the bundled SQLite re-prepare a cached statement.
//! Re-preparing on every run costs a full parse and plan; the store's hot paths must avoid it.
use rusqlite::{params, Connection, StatementStatus};

fn db() -> Connection {
    let db = Connection::open_in_memory().unwrap();
    db.execute_batch(
        "CREATE TABLE t_words(id INTEGER PRIMARY KEY, c_word TEXT); CREATE INDEX t_words_c ON t_words(c_word);
         CREATE TABLE edges(src INTEGER NOT NULL,rel TEXT NOT NULL,dst INTEGER NOT NULL,weight REAL NOT NULL,PRIMARY KEY(src,rel,dst));
         CREATE INDEX edges_in ON edges(dst,rel,src);",
    )
    .unwrap();
    let tx = db.unchecked_transaction().unwrap();
    for id in 0..5000i64 {
        tx.execute(
            "INSERT INTO t_words VALUES(?,?)",
            params![id, format!("w{}", id % 500)],
        )
        .unwrap();
        tx.execute(
            "INSERT INTO edges VALUES(?,'r',?,1.0)",
            params![id % 700, id],
        )
        .unwrap();
    }
    tx.commit().unwrap();
    db
}

/// Runs `sql` 200 times through the statement cache and returns (reprepares, sorts).
/// `per_call` takes the statement from the cache on every run, as the store does; returning it
/// to the cache clears its bindings.
fn count(
    db: &Connection,
    sql: &str,
    text_key: bool,
    limit_value: bool,
    per_call: bool,
) -> (i64, i64) {
    if per_call {
        for _ in 0..200 {
            let mut stmt = db.prepare_cached(sql).unwrap();
            let key: rusqlite::types::Value = if text_key {
                "w7".to_string().into()
            } else {
                7i64.into()
            };
            let mut rows = if limit_value {
                stmt.query(params![key, 100i64]).unwrap()
            } else {
                stmt.query(params![key]).unwrap()
            };
            while rows.next().unwrap().is_some() {}
        }
    }
    let mut stmt = db.prepare_cached(sql).unwrap();
    if per_call {
        return (
            stmt.get_status(StatementStatus::RePrepare) as i64,
            stmt.get_status(StatementStatus::Sort) as i64,
        );
    }
    for _ in 0..200 {
        let key: rusqlite::types::Value = if text_key {
            "w7".to_string().into()
        } else {
            7i64.into()
        };
        let mut rows = if limit_value {
            stmt.query(params![key, 100i64]).unwrap()
        } else {
            stmt.query(params![key]).unwrap()
        };
        while rows.next().unwrap().is_some() {}
    }
    (
        stmt.get_status(StatementStatus::RePrepare) as i64,
        stmt.get_status(StatementStatus::Sort) as i64,
    )
}

#[test]
fn reprepare_forms() {
    let db = db();
    let version: String = db
        .query_row("SELECT sqlite_version()", [], |r| r.get(0))
        .unwrap();
    println!("REPREPARE sqlite_version={version}");
    let mut opts = db.prepare("PRAGMA compile_options").unwrap();
    let opts: Vec<String> = opts
        .query_map([], |r| r.get(0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    println!("REPREPARE compile_options={}", opts.join(","));
    let eq = "SELECT id,c_word FROM t_words WHERE c_word=?1 ORDER BY id";
    let ed = "SELECT src,rel,dst,weight FROM edges WHERE src=?1 ORDER BY rel,src,dst";
    for phase in ["fresh", "analyzed", "analyzed_qpsg"] {
        if phase == "analyzed" {
            db.execute_batch("ANALYZE").unwrap();
            db.flush_prepared_statement_cache();
        }
        if phase == "analyzed_qpsg" {
            db.set_db_config(
                rusqlite::config::DbConfig::SQLITE_DBCONFIG_ENABLE_QPSG,
                true,
            )
            .unwrap();
            db.flush_prepared_statement_cache();
        }
        for (label, base, text) in [("eq", eq, true), ("edges", ed, false)] {
            for (form, tail, lim) in [
                ("literal", " LIMIT 100", false),
                ("bare", " LIMIT ?2", true),
                ("cast", " LIMIT CAST(?2 AS INTEGER)", true),
                ("plus0", " LIMIT (?2+0)", true),
                ("subselect", " LIMIT (SELECT ?2)", true),
                ("nolimit", "", false),
            ] {
                let sql = format!("{base}{tail}");
                db.flush_prepared_statement_cache();
                let (re, so) = count(&db, &sql, text, lim, false);
                println!(
                    "REPREPARE {phase} held {label}_{form} runs=200 reprepares={re} sorts={so}"
                );
                db.flush_prepared_statement_cache();
                let (re, so) = count(&db, &sql, text, lim, true);
                println!(
                    "REPREPARE {phase} per_call {label}_{form} runs=200 reprepares={re} sorts={so}"
                );
                // The store's configuration (QPSG on) and its LIMIT form, after ANALYZE as the worst case.
                if phase == "analyzed_qpsg" && form == "cast" {
                    assert_eq!(
                        (re, so),
                        (0, 0),
                        "{label}: the store's statement form re-prepares or sorts"
                    );
                }
            }
        }
    }
}
