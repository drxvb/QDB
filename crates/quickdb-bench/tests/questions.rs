use quickdb_bench::questions::QuestionSet;
use quickdb_bench::{corpus, run, run_with_questions, MemStore};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
const DIM: usize = 4;

fn scratch(label: &str) -> PathBuf {
    let base = std::env::var_os("QUICKDB_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join(format!(
        "questions-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Unit vector at `degrees` in the first plane, so every distance order below is known by hand.
fn at(degrees: f32) -> Vec<f32> {
    let r = degrees.to_radians();
    vec![r.cos(), r.sin(), 0.0, 0.0]
}
fn write_f32(path: &Path, rows: &[Vec<f32>]) {
    let bytes: Vec<u8> = rows
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(path, bytes).expect("write vectors");
}
fn write_lines(path: &Path, lines: &[serde_json::Value]) {
    let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
    std::fs::write(path, text).expect("write jsonl");
}

/// Six documents. Document 5 has no chunk. Chunks (id, doc, angle):
/// 100,101 doc 1 at 0, 10 | 102 doc 2 at 90 | 103,104 doc 3 at 100, 110 | 105 doc 4 at 180 | 106..=111 doc 6 at 120..=170.
fn make_corpus(dir: &Path) {
    std::fs::create_dir_all(dir).expect("mkdir");
    let layout: [(u64, u64, f32); 12] = [
        (100, 1, 0.0),
        (101, 1, 10.0),
        (102, 2, 90.0),
        (103, 3, 100.0),
        (104, 3, 110.0),
        (105, 4, 180.0),
        (106, 6, 120.0),
        (107, 6, 130.0),
        (108, 6, 140.0),
        (109, 6, 150.0),
        (110, 6, 160.0),
        (111, 6, 170.0),
    ];
    let chunks: Vec<_> = layout
        .iter()
        .map(|(id, doc, _)| {
            let text = format!("chunk {id} العربية");
            serde_json::json!({"id": id, "doc_id": doc, "corpus": if *doc <= 3 { "alpha" } else { "beta" },
                "path": format!("d{doc}.md"), "seq": 0, "chars": text.chars().count(), "text": text})
        })
        .collect();
    write_lines(&dir.join("chunks.jsonl"), &chunks);
    let docs: Vec<_> = (1..=6)
        .map(|d| serde_json::json!({"id": d, "corpus": "alpha", "path": format!("d{d}.md"), "chunks": 1, "chars": 10}))
        .collect();
    write_lines(&dir.join("docs.jsonl"), &docs);
    write_lines(
        &dir.join("edges.jsonl"),
        &[serde_json::json!({"src": 100, "rel": "next", "dst": 101, "weight": 1.0})],
    );
    let vectors: Vec<_> = layout.iter().map(|(_, _, a)| at(*a)).collect();
    write_f32(&dir.join("vectors.f32"), &vectors);
    std::fs::write(
        dir.join("meta.json"),
        serde_json::json!({"chunks": 12, "docs": 6, "edges": 1, "dim": DIM}).to_string(),
    )
    .expect("meta");
}

/// (id, lang, corpus, expected doc ids, query angle). Expected results are worked out in `known_answers`.
fn make_questions(dir: &Path) {
    std::fs::create_dir_all(dir).expect("mkdir");
    let rows: [(&str, &str, &str, Vec<u64>, f32); 8] = [
        ("q001", "en", "alpha", vec![1], 8.0),
        ("q002", "en", "alpha", vec![4], 8.0),
        ("q003", "ar", "alpha", vec![2], 96.0),
        ("q004", "ar", "beta", vec![4], 96.0),
        ("q005", "en", "beta", vec![999], 8.0),
        ("q006", "ar", "beta", vec![1000], 96.0),
        ("q007", "en", "beta", vec![5], 8.0),
        ("q008", "ar", "beta", vec![4, 999], 176.0),
    ];
    let lines: Vec<_> = rows
        .iter()
        .map(|(id, lang, corpus, expected, _)| {
            serde_json::json!({"id": id, "lang": lang, "question": "placeholder", "expected_doc_ids": expected, "corpus": corpus})
        })
        .collect();
    write_lines(&dir.join("questions.jsonl"), &lines);
    let vectors: Vec<_> = rows.iter().map(|r| at(r.4)).collect();
    write_f32(&dir.join("qvectors.f32"), &vectors);
}

fn close(value: &serde_json::Value, want: f64) {
    let got = value.as_f64().expect("number");
    assert!((got - want).abs() < 1e-9, "{got} != {want}");
}

#[test]
fn known_answers() {
    // From 8 degrees the ten nearest chunks leave out 105 (doc 4) and 111, so q002 misses.
    // From 96 degrees the distinct documents arrive as 3, 2, 6, 4, so q003 ranks 2nd and q004 ranks 4th.
    // q005 and q006 name documents that do not exist, q007 names a document with no chunk, q008 keeps doc 4 only.
    let root = scratch("known");
    let (corpus_dir, questions_dir) = (root.join("corpus"), root.join("questions"));
    make_corpus(&corpus_dir);
    make_questions(&questions_dir);
    let data = corpus(&corpus_dir).expect("corpus");
    let set = QuestionSet::load(&corpus_dir, &questions_dir, data.dim).expect("questions");
    let report = run_with_questions::<MemStore>(&data, &root.join("db"), Some(&set)).expect("run");
    let block = report.questions.expect("questions block");
    assert_eq!(block["count"], 6);
    assert_eq!(block["unresolved"], 2);
    assert_eq!(
        block["by_lang"]["en"],
        serde_json::json!({"count": 3, "unresolved": 1})
    );
    assert_eq!(
        block["by_lang"]["ar"],
        serde_json::json!({"count": 3, "unresolved": 1})
    );
    assert_eq!(
        block["by_corpus"]["alpha"],
        serde_json::json!({"count": 3, "unresolved": 0})
    );
    assert_eq!(
        block["by_corpus"]["beta"],
        serde_json::json!({"count": 3, "unresolved": 2})
    );
    // MemStore is exact, so every setting must agree with the exact entry.
    for name in ["exact", "knn_ef32", "knn_ef64", "knn_ef128", "knn_ef256"] {
        let r = &block["results"][name];
        assert_eq!(r["count"], 6, "{name}");
        close(&r["doc_hit_at_10"], 4.0 / 6.0);
        close(&r["mrr_at_10"], 2.75 / 6.0);
        close(&r["recall_at_10_vs_exact"], 1.0);
        assert!(r["p50_us"].as_f64().expect("p50") >= 0.0);
        close(&r["by_lang"]["en"]["doc_hit_at_10"], 1.0 / 3.0);
        close(&r["by_lang"]["en"]["mrr_at_10"], 1.0 / 3.0);
        close(&r["by_lang"]["ar"]["doc_hit_at_10"], 1.0);
        close(&r["by_lang"]["ar"]["mrr_at_10"], 1.75 / 3.0);
        close(&r["by_corpus"]["alpha"]["doc_hit_at_10"], 2.0 / 3.0);
        close(&r["by_corpus"]["alpha"]["mrr_at_10"], 0.5);
        close(&r["by_corpus"]["beta"]["doc_hit_at_10"], 2.0 / 3.0);
        close(&r["by_corpus"]["beta"]["mrr_at_10"], 1.25 / 3.0);
        assert_eq!(r["by_lang"]["ar"]["count"], 3);
    }
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn old_style_run_keeps_its_keys() {
    let root = scratch("keys");
    let (corpus_dir, questions_dir) = (root.join("corpus"), root.join("questions"));
    make_corpus(&corpus_dir);
    make_questions(&questions_dir);
    let data = corpus(&corpus_dir).expect("corpus");
    let keys = |v: &serde_json::Value| -> Vec<String> {
        let mut k: Vec<String> = v.as_object().expect("object").keys().cloned().collect();
        k.sort();
        k
    };
    let old = serde_json::to_value(run::<MemStore>(&data, &root.join("old")).expect("old run"))
        .expect("json");
    let mut expected: Vec<String> = [
        "schema_version",
        "backend",
        "dataset",
        "hostname",
        "cpu_count",
        "available_parallelism",
        "phases_seconds",
        "size_bytes",
        "memory",
        "queries",
        "rows",
        "vectors",
        "edges",
        "metric",
        "query_seed",
    ]
    .map(String::from)
    .to_vec();
    expected.sort();
    assert_eq!(keys(&old), expected);
    let none = run_with_questions::<MemStore>(&data, &root.join("none"), None).expect("run");
    assert_eq!(keys(&serde_json::to_value(none).expect("json")), expected);
    let set = QuestionSet::load(&corpus_dir, &questions_dir, data.dim).expect("questions");
    let new = serde_json::to_value(
        run_with_questions::<MemStore>(&data, &root.join("new"), Some(&set)).expect("run"),
    )
    .expect("json");
    let mut with = expected.clone();
    with.push("questions".into());
    with.sort();
    assert_eq!(keys(&new), with);
    assert_eq!(keys(&new["queries"]), keys(&old["queries"]));
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn bad_question_inputs_are_schema_errors_without_content() {
    let root = scratch("bad");
    let (corpus_dir, questions_dir) = (root.join("corpus"), root.join("questions"));
    make_corpus(&corpus_dir);
    make_questions(&questions_dir);
    let load = || QuestionSet::load(&corpus_dir, &questions_dir, DIM);
    assert!(load().is_ok());
    let vectors = questions_dir.join("qvectors.f32");
    let good = std::fs::read(&vectors).expect("read");
    std::fs::write(&vectors, &good[..good.len() - 4]).expect("short");
    assert!(load().is_err());
    let mut nan = good.clone();
    nan[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    std::fs::write(&vectors, nan).expect("nan");
    assert!(load().is_err());
    std::fs::write(&vectors, &good).expect("restore");
    std::fs::write(
        questions_dir.join("qmeta.json"),
        r#"{"count": 7, "dim": 4}"#,
    )
    .expect("meta");
    assert!(load().is_err());
    std::fs::write(
        questions_dir.join("qmeta.json"),
        r#"{"count": 8, "dim": 4}"#,
    )
    .expect("meta");
    assert!(load().is_ok());
    std::fs::write(
        questions_dir.join("questions.jsonl"),
        "SECRET-TEXT not json\n",
    )
    .expect("bad json");
    let message = load().err().expect("error").to_string();
    assert!(
        message.contains("questions.jsonl:1") && !message.contains("SECRET"),
        "{message}"
    );
    std::fs::remove_dir_all(&root).ok();
}
