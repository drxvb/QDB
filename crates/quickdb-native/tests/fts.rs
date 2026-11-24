//! Full-text and hybrid search on `NativeStore`.
use quickdb_api::*;
use quickdb_native::{fts, NativeStore, TextIndexDef};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn data_root() -> PathBuf {
    std::env::var_os("QUICKDB_DATA").map_or_else(std::env::temp_dir, PathBuf::from)
}

fn tempdir() -> tempfile::TempDir {
    tempfile::tempdir_in(data_root()).unwrap()
}

fn open(dir: &Path) -> NativeStore {
    NativeStore::open(
        dir,
        &OpenOptions {
            create: true,
            sync_each_write: false,
        },
    )
    .unwrap()
}

fn col(name: &str, ty: ColumnType) -> ColumnDef {
    ColumnDef {
        name: name.into(),
        ty,
        indexed: false,
    }
}

fn notes_table() -> TableDef {
    TableDef {
        name: "notes".into(),
        columns: vec![
            col("title", ColumnType::Text),
            col("body", ColumnType::Text),
            col("score", ColumnType::Int),
        ],
    }
}

fn notes_index() -> TextIndexDef {
    TextIndexDef {
        name: "notes_text".into(),
        table: "notes".into(),
        columns: vec!["title".into(), "body".into()],
    }
}

fn note(id: NodeId, title: &str, body: &str) -> Row {
    Row {
        id,
        values: vec![
            Value::Text(title.into()),
            Value::Text(body.into()),
            Value::Int(id as i64),
        ],
    }
}

fn ids(hits: &[(NodeId, f64)]) -> Vec<NodeId> {
    hits.iter().map(|h| h.0).collect()
}

fn search(s: &NativeStore, query: &str) -> Vec<NodeId> {
    ids(&s.text_search("notes_text", query, 100).unwrap())
}

fn sorted(mut ids: Vec<NodeId>) -> Vec<NodeId> {
    ids.sort_unstable();
    ids
}

fn store_with_notes(dir: &Path) -> NativeStore {
    let mut s = open(dir);
    s.define_table(&notes_table()).unwrap();
    s.define_text_index(&notes_index()).unwrap();
    s
}

#[test]
fn english_search_ranks_and_requires_every_term() {
    let dir = tempdir();
    let mut s = store_with_notes(dir.path());
    s.upsert_rows(
        "notes",
        &[
            note(
                1,
                "Glacier survey",
                "The glacier retreated; glacier ice thinned.",
            ),
            note(2, "Harbour log", "A ship left the harbour at dawn."),
            note(
                3,
                "Field notes",
                "One glacier photo among many harbour photos.",
            ),
            note(4, "Café menu", "Crème brûlée and coffee."),
        ],
    )
    .unwrap();
    let hits = s.text_search("notes_text", "GLACIER", 10).unwrap();
    assert_eq!(ids(&hits), [1, 3]);
    assert!(hits[0].1 > hits[1].1 && hits[1].1 > 0.0);
    assert_eq!(search(&s, "glacier harbour"), [3]);
    assert_eq!(search(&s, "glacier submarine"), Vec::<NodeId>::new());
    assert_eq!(search(&s, "cafe creme"), [4]);
    assert_eq!(search(&s, "Survey"), [1]);
    assert_eq!(s.text_search("notes_text", "glacier", 1).unwrap().len(), 1);
    assert!(s
        .text_search("notes_text", "glacier", 0)
        .unwrap()
        .is_empty());
    assert!(matches!(
        s.text_search("missing", "glacier", 10),
        Err(Error::NotFound(_))
    ));
}

#[test]
fn arabic_search_ignores_harakat_tatweel_and_alef_forms() {
    let dir = tempdir();
    let mut s = store_with_notes(dir.path());
    let voweled = "قَرَأَ الطّالِبُ الدَّرْسَ";
    s.upsert_rows(
        "notes",
        &[
            note(1, "درس", voweled),
            note(2, "رحلة", "سافر أحمد إلى الجبل"),
            note(3, "مكتبة", "كـــتاب عن الإسلام والآثار"),
            note(4, "رسالة", "كتب إبراهيم رسالة طويلة"),
        ],
    )
    .unwrap();
    assert_eq!(search(&s, "قرأ الطالب"), [1]);
    assert_eq!(search(&s, "قرا الدرس"), [1]);
    assert_eq!(search(&s, "الطَّالِبُ"), [1]);
    for spelling in ["احمد", "أحمد", "إحمد", "آحمد"] {
        assert_eq!(search(&s, spelling), [2], "{spelling}");
    }
    assert_eq!(search(&s, "الي"), [2]);
    assert_eq!(search(&s, "إلى"), [2]);
    assert_eq!(search(&s, "كتاب"), [3]);
    assert_eq!(search(&s, "الإسلام والاثار"), [3]);
    assert_eq!(search(&s, "ابراهيم"), [4]);
    assert_eq!(search(&s, "رساله"), [4]);
    assert_eq!(search(&s, "رحلة"), [2]);
    assert_eq!(
        s.get_row("notes", 1).unwrap().unwrap().values[1],
        Value::Text(voweled.into())
    );
}

#[test]
fn updates_and_deletes_stay_in_sync() {
    let dir = tempdir();
    let mut s = store_with_notes(dir.path());
    s.upsert_rows(
        "notes",
        &[note(1, "red", "apple"), note(2, "green", "apple")],
    )
    .unwrap();
    assert_eq!(search(&s, "apple"), [1, 2]);
    s.upsert_rows("notes", &[note(1, "yellow", "pear")])
        .unwrap();
    assert_eq!(search(&s, "apple"), [2]);
    assert_eq!(search(&s, "red"), Vec::<NodeId>::new());
    assert_eq!(search(&s, "yellow pear"), [1]);
    s.upsert_rows(
        "notes",
        &[Row {
            id: 2,
            values: vec![Value::Null, Value::Text("plum".into()), Value::Null],
        }],
    )
    .unwrap();
    assert_eq!(search(&s, "green"), Vec::<NodeId>::new());
    assert_eq!(search(&s, "plum"), [2]);
    assert_eq!(s.delete_rows("notes", &[1, 99]).unwrap(), 1);
    assert_eq!(search(&s, "pear"), Vec::<NodeId>::new());
    assert_eq!(search(&s, "plum"), [2]);
    s.upsert_rows("notes", &[note(1, "red", "apple")]).unwrap();
    assert_eq!(search(&s, "apple"), [1]);
    let bad = Row {
        id: 3,
        values: vec![Value::Int(1), Value::Null, Value::Null],
    };
    assert!(s
        .upsert_rows("notes", &[note(4, "kiwi", "kiwi"), bad])
        .is_err());
    assert_eq!(search(&s, "kiwi"), Vec::<NodeId>::new());
}

#[test]
fn defining_an_index_on_existing_rows_indexes_them() {
    let dir = tempdir();
    let mut s = open(dir.path());
    s.define_table(&notes_table()).unwrap();
    let rows: Vec<_> = (1..=300)
        .map(|i| {
            note(
                i,
                &format!("entry {i}"),
                if i % 3 == 0 { "fizz" } else { "plain" },
            )
        })
        .collect();
    s.upsert_rows("notes", &rows).unwrap();
    s.upsert_rows(
        "notes",
        &[Row {
            id: 301,
            values: vec![Value::Null, Value::Null, Value::Null],
        }],
    )
    .unwrap();
    s.define_text_index(&notes_index()).unwrap();
    s.define_text_index(&notes_index()).unwrap();
    assert_eq!(search(&s, "fizz").len(), 100);
    assert_eq!(search(&s, "entry 150"), [150]);
    assert_eq!(s.text_indexes(), vec![notes_index()]);
    let mut other = notes_index();
    other.columns = vec!["body".into()];
    assert!(matches!(s.define_text_index(&other), Err(Error::Schema(_))));
    let cases = [
        ("missing_table", "nope", vec!["body"]),
        ("missing_column", "notes", vec!["nope"]),
        ("not_text", "notes", vec!["score"]),
        ("duplicate", "notes", vec!["body", "body"]),
        ("no_columns", "notes", vec![]),
        ("Bad Name", "notes", vec!["body"]),
    ];
    for (name, table, columns) in cases {
        let def = TextIndexDef {
            name: name.into(),
            table: table.into(),
            columns: columns.into_iter().map(String::from).collect(),
        };
        assert!(s.define_text_index(&def).is_err(), "{name}");
    }
    assert_eq!(s.text_indexes().len(), 1);
}

#[test]
fn two_indexes_with_overlapping_names_and_columns() {
    let dir = tempdir();
    let mut s = store_with_notes(dir.path());
    for (name, column) in [("t", "title"), ("t_data", "body")] {
        s.define_text_index(&TextIndexDef {
            name: name.into(),
            table: "notes".into(),
            columns: vec![column.into()],
        })
        .unwrap();
    }
    s.upsert_rows(
        "notes",
        &[note(1, "orbit", "comet"), note(2, "comet", "orbit")],
    )
    .unwrap();
    assert_eq!(ids(&s.text_search("t", "orbit", 10).unwrap()), [1]);
    assert_eq!(ids(&s.text_search("t_data", "orbit", 10).unwrap()), [2]);
    assert_eq!(search(&s, "orbit"), [1, 2]);
    s.delete_rows("notes", &[1]).unwrap();
    assert_eq!(
        ids(&s.text_search("t", "orbit", 10).unwrap()),
        Vec::<NodeId>::new()
    );
}

#[test]
fn reopen_after_flush_and_after_drop() {
    let dir = tempdir();
    {
        let mut s = store_with_notes(dir.path());
        s.upsert_rows("notes", &[note(1, "lantern", "oil")])
            .unwrap();
        s.flush().unwrap();
        s.upsert_rows("notes", &[note(2, "lantern", "wick")])
            .unwrap();
    }
    {
        let mut s = open(dir.path());
        assert_eq!(s.text_indexes(), vec![notes_index()]);
        assert_eq!(search(&s, "lantern"), [1, 2]);
        s.upsert_rows("notes", &[note(1, "candle", "wax")]).unwrap();
        s.delete_rows("notes", &[2]).unwrap();
        s.flush().unwrap();
    }
    let s = open(dir.path());
    assert_eq!(search(&s, "lantern"), Vec::<NodeId>::new());
    assert_eq!(search(&s, "candle wax"), [1]);
}

#[test]
fn read_only_handle_searches_live_text() {
    let dir = tempdir();
    let mut w = store_with_notes(dir.path());
    w.upsert_rows("notes", &[note(1, "meadow", "clover")])
        .unwrap();
    w.flush().unwrap();
    let mut r = NativeStore::open_read_only(dir.path()).unwrap();
    assert_eq!(search(&r, "clover"), [1]);
    w.upsert_rows("notes", &[note(2, "meadow", "thistle")])
        .unwrap();
    assert_eq!(search(&r, "meadow"), [1, 2]);
    assert!(matches!(
        r.define_text_index(&TextIndexDef {
            name: "other".into(),
            table: "notes".into(),
            columns: vec!["body".into()],
        }),
        Err(Error::Backend(_))
    ));
    drop(w);
    drop(r);
    let r = NativeStore::open_read_only(dir.path()).unwrap();
    assert_eq!(search(&r, "thistle"), [2]);
}

#[test]
fn hostile_queries_are_plain_text() {
    let dir = tempdir();
    let mut s = store_with_notes(dir.path());
    s.upsert_rows(
        "notes",
        &[
            note(1, "quick fox", "near the river"),
            note(2, "slow fox", "title body"),
        ],
    )
    .unwrap();
    let long_words: String = (0..20_000).map(|i| format!("w{i} ")).collect();
    let long_token = "x".repeat(1 << 20);
    let hostile = [
        "\"",
        "\"\"\"",
        "\"fox",
        "fox\"",
        "NEAR(",
        "NEAR(quick fox)",
        "NEAR(quick fox, 2)",
        "*",
        "fox*",
        "-fox",
        "- -",
        "title:fox",
        "c_title:fox",
        "c0:fox",
        "{c0 c1}: fox",
        ":",
        "fox OR",
        "AND",
        "NOT fox",
        ")(",
        "^fox",
        "{title body}: fox",
        "fox\0quick",
        "\u{202e}fox",
        long_words.as_str(),
        long_token.as_str(),
    ];
    for q in hostile {
        let shown: String = q.chars().take(40).collect();
        assert!(s.text_search("notes_text", q, 10).is_ok(), "{shown:?}");
    }
    assert_eq!(sorted(search(&s, "fox*")), [1, 2]);
    assert_eq!(sorted(search(&s, "-fox")), [1, 2]);
    assert_eq!(sorted(search(&s, "\"fox")), [1, 2]);
    assert_eq!(search(&s, "NEAR(quick fox)"), [1]);
    assert_eq!(search(&s, "fox OR"), Vec::<NodeId>::new());
    assert_eq!(search(&s, "title:fox"), [2]);
    for empty in ["", "   ", "\"", "*", ":", "- -", ")(", "!?.,;"] {
        assert!(
            s.text_search("notes_text", empty, 10).unwrap().is_empty(),
            "{empty:?}"
        );
    }
}

#[test]
fn hybrid_search_fuses_disagreeing_rankings() {
    let dir = tempdir();
    let mut s = store_with_notes(dir.path());
    let texts = [
        (1, "alpha alpha alpha"),
        (2, "alpha alpha beta gamma"),
        (3, "alpha beta gamma delta epsilon zeta"),
        (4, "omega"),
        (5, "sigma"),
    ];
    let mut rows: Vec<_> = texts.iter().map(|&(id, body)| note(id, "", body)).collect();
    rows.extend((6..=12).map(|id| note(id, "", "filler words only")));
    s.upsert_rows("notes", &rows).unwrap();
    s.define_vector_index(&VectorIndexDef {
        name: "emb".into(),
        dim: 2,
        metric: Metric::L2,
        m: 8,
        ef_construction: 32,
    })
    .unwrap();
    let vectors: Vec<(NodeId, [f32; 2])> = vec![
        (4, [0.1, 0.0]),
        (2, [0.2, 0.0]),
        (5, [0.3, 0.0]),
        (3, [0.4, 0.0]),
        (1, [0.5, 0.0]),
    ];
    let items: Vec<_> = vectors.iter().map(|(id, v)| (*id, &v[..])).collect();
    s.upsert_vectors("emb", &items).unwrap();
    let query = [0.0f32, 0.0];
    assert_eq!(search(&s, "alpha"), [1, 2, 3]);
    let knn: Vec<_> = s
        .knn("emb", &query, 5, 32)
        .unwrap()
        .iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(knn, [4, 2, 5, 3, 1]);
    let fused = s
        .hybrid_search("notes_text", "emb", "alpha", &query, 5, 32)
        .unwrap();
    assert_eq!(ids(&fused), [2, 1, 3, 4, 5]);
    let rrf = |r: f64| 1.0 / (fts::RRF_K + r);
    let expected = [
        rrf(2.0) + rrf(2.0),
        rrf(1.0) + rrf(5.0),
        rrf(3.0) + rrf(4.0),
        rrf(1.0),
        rrf(3.0),
    ];
    for (got, want) in fused.iter().zip(expected) {
        assert!((got.1 - want).abs() < 1e-12, "{got:?} vs {want}");
    }
    let top = s
        .hybrid_search("notes_text", "emb", "alpha", &query, 2, 32)
        .unwrap();
    assert_eq!(ids(&top), [2, 1]);
    let vector_only = s
        .hybrid_search("notes_text", "emb", "\"*", &query, 3, 32)
        .unwrap();
    assert_eq!(ids(&vector_only), [4, 2, 5]);
    assert!(s
        .hybrid_search("notes_text", "emb", "alpha", &query, 0, 32)
        .unwrap()
        .is_empty());
    assert!(matches!(
        s.hybrid_search("nope", "emb", "alpha", &query, 3, 32),
        Err(Error::NotFound(_))
    ));
    assert!(matches!(
        s.hybrid_search("notes_text", "nope", "alpha", &query, 3, 32),
        Err(Error::NotFound(_))
    ));
    assert!(matches!(
        s.hybrid_search("notes_text", "emb", "alpha", &[0.0], 3, 32),
        Err(Error::Schema(_))
    ));
}

const CHILD_ENV: &str = "QUICKDB_FTS_CRASH_DIR";
const BATCH: u64 = 40;

fn crash_text(id: NodeId, generation: u64) -> String {
    format!("w{id} gen{generation} common")
}

/// Writer process for `killed_writer_leaves_text_and_rows_consistent`; a no-op otherwise.
#[test]
fn crash_child_writer() {
    let Some(dir) = std::env::var_os(CHILD_ENV) else {
        return;
    };
    let mut s = open(Path::new(&dir));
    s.define_table(&TableDef {
        name: "notes".into(),
        columns: vec![col("body", ColumnType::Text)],
    })
    .unwrap();
    s.define_text_index(&TextIndexDef {
        name: "notes_text".into(),
        table: "notes".into(),
        columns: vec!["body".into()],
    })
    .unwrap();
    for n in 0.. {
        let mut rows: Vec<_> = (0..BATCH)
            .map(|i| n * BATCH + i)
            .map(|id| Row {
                id,
                values: vec![Value::Text(crash_text(id, n))],
            })
            .collect();
        rows.extend((0..n * BATCH).step_by(7).map(|id| Row {
            id,
            values: vec![Value::Text(crash_text(id, n))],
        }));
        s.upsert_rows("notes", &rows).unwrap();
        let deleted: Vec<_> = (0..n * BATCH).skip(n as usize % 5).step_by(11).collect();
        s.delete_rows("notes", &deleted).unwrap();
        println!("COMMITTED {n}");
    }
}

#[test]
fn killed_writer_leaves_text_and_rows_consistent() {
    for (run, wait_for) in [1u64, 6, 25].into_iter().enumerate() {
        let dir = tempdir();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "crash_child_writer",
                "--exact",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD_ENV, dir.path())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
        let mut committed = 0;
        while committed < wait_for {
            let line = lines.next().expect("writer exited early").unwrap();
            if line.starts_with("COMMITTED ") {
                committed += 1;
            }
        }
        std::thread::sleep(std::time::Duration::from_micros(300 * run as u64));
        child.kill().unwrap();
        child.wait().unwrap();
        drop(lines);

        let s = open(dir.path());
        let rows = s.select("notes", &Filter::All, usize::MAX).unwrap();
        assert!(rows.len() as u64 >= wait_for * BATCH / 2);
        let mut by_generation: BTreeMap<u64, BTreeSet<NodeId>> = BTreeMap::new();
        for row in &rows {
            let Value::Text(body) = &row.values[0] else {
                panic!("row {} lost its text", row.id);
            };
            let generation: u64 = body.split(' ').nth(1).unwrap()[3..].parse().unwrap();
            assert_eq!(*body, crash_text(row.id, generation));
            by_generation.entry(generation).or_default().insert(row.id);
            let own = ids(&s
                .text_search("notes_text", &format!("w{}", row.id), 10)
                .unwrap());
            assert_eq!(own, [row.id]);
        }
        let all: BTreeSet<NodeId> = rows.iter().map(|r| r.id).collect();
        let common: BTreeSet<NodeId> =
            ids(&s.text_search("notes_text", "common", usize::MAX).unwrap())
                .into_iter()
                .collect();
        assert_eq!(common, all, "run {run}: text index and rows disagree");
        let last = by_generation.keys().copied().max().unwrap();
        for generation in 0..=last + 1 {
            let found: BTreeSet<NodeId> = ids(&s
                .text_search("notes_text", &format!("gen{generation}"), usize::MAX)
                .unwrap())
            .into_iter()
            .collect();
            assert_eq!(
                found,
                by_generation.remove(&generation).unwrap_or_default(),
                "run {run}: generation {generation}"
            );
        }
        println!(
            "FTS_CRASH run={run} waited_for={wait_for} rows={} last_generation={last} consistent=true",
            rows.len()
        );
    }
}

#[test]
fn any_word_search_answers_natural_questions() {
    let dir = tempdir();
    let mut s = store_with_notes(dir.path());
    s.upsert_rows(
        "notes",
        &[
            note(1, "", "store notes locally in one embedded database"),
            note(2, "", "tablets often have no network"),
            note(3, "", "weekly status meeting moved to Monday"),
        ],
    )
    .unwrap();
    let question = "how do we store notes without network";
    // Every word: no single note contains them all.
    assert!(s
        .text_search("notes_text", question, 10)
        .unwrap()
        .is_empty());
    // Any word: notes 1 and 2 match, note 3 does not; more matching words rank higher.
    let any = s.text_search_any("notes_text", question, 10).unwrap();
    assert_eq!(ids(&any), [1, 2]);
    assert!(any[0].1 >= any[1].1);
    // Hybrid search uses any-word matching for its text ranking.
    s.define_vector_index(&VectorIndexDef {
        name: "emb".into(),
        dim: 2,
        metric: Metric::L2,
        m: 8,
        ef_construction: 32,
    })
    .unwrap();
    let v: Vec<(NodeId, [f32; 2])> = vec![(1, [1.0, 0.0]), (2, [0.0, 1.0]), (3, [5.0, 5.0])];
    let items: Vec<_> = v.iter().map(|(id, x)| (*id, &x[..])).collect();
    s.upsert_vectors("emb", &items).unwrap();
    let fused = s
        .hybrid_search("notes_text", "emb", question, &[0.9, 0.1], 3, 16)
        .unwrap();
    assert_eq!(fused[0].0, 1);
    assert_eq!(fused.last().unwrap().0, 3);
}
