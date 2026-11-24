//! Agent memory in one embedded store: remember notes, recall them by words and by meaning, and
//! follow how they relate.
//!
//! Each memory is a row (kind + text), a vector (its embedding) and a graph node (links such as
//! "follows" or "about"). Recall fuses full-text BM25 and vector search with Reciprocal Rank
//! Fusion, then walks the graph from the best hit.
//!
//! The embedding here is a toy (hashed bag of words) so the example runs with no model. A real
//! agent passes the vectors of its embedding model instead; nothing else changes.
//!
//! Run: `cargo run --release -p quickdb --example agent_memory`

use std::path::PathBuf;

use quickdb::{
    ColumnDef, ColumnType, Direction, Edge, Filter, Metric, NativeStore, OpenOptions, Row, Store,
    TableDef, TextIndexDef, Value, VectorIndexDef,
};

const DIM: usize = 64;

/// Toy embedding: hashed bag of words, L2-normalised. Stand-in for a real embedding model.
fn embed(text: &str) -> Vec<f32> {
    let mut v = vec![0f32; DIM];
    for word in text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
    {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in word.to_lowercase().bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        v[(h % DIM as u64) as usize] += 1.0;
    }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    v.iter().map(|x| x / n).collect()
}

struct Memory {
    db: NativeStore,
    next_id: u64,
}

impl Memory {
    fn open(dir: &std::path::Path) -> quickdb::Result<Self> {
        let mut db = NativeStore::open(dir, &OpenOptions::default())?;
        db.define_table(&TableDef {
            name: "memories".into(),
            columns: vec![
                ColumnDef {
                    name: "kind".into(),
                    ty: ColumnType::Text,
                    indexed: true,
                },
                ColumnDef {
                    name: "text".into(),
                    ty: ColumnType::Text,
                    indexed: false,
                },
            ],
        })?;
        db.define_vector_index(&VectorIndexDef {
            name: "emb".into(),
            dim: DIM,
            metric: Metric::Cosine,
            m: 16,
            ef_construction: 200,
        })?;
        db.define_text_index(&TextIndexDef {
            name: "words".into(),
            table: "memories".into(),
            columns: vec!["text".into()],
        })?;
        let next_id = db.count_rows("memories")? + 1;
        Ok(Memory { db, next_id })
    }

    /// Store one memory; optionally link it to an earlier one.
    fn remember(
        &mut self,
        kind: &str,
        text: &str,
        link: Option<(&str, u64)>,
    ) -> quickdb::Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        self.db.upsert_rows(
            "memories",
            &[Row {
                id,
                values: vec![Value::Text(kind.into()), Value::Text(text.into())],
            }],
        )?;
        self.db
            .upsert_vectors("emb", &[(id, embed(text).as_slice())])?;
        if let Some((rel, to)) = link {
            self.db.upsert_edges(&[Edge {
                src: id,
                rel: rel.into(),
                dst: to,
                weight: 1.0,
            }])?;
        }
        Ok(id)
    }

    /// Best memories for a question: words and meaning fused, best first.
    fn recall(&self, question: &str, k: usize) -> quickdb::Result<Vec<(u64, f64)>> {
        self.db
            .hybrid_search("words", "emb", question, &embed(question), k, 64)
    }

    fn text(&self, id: u64) -> quickdb::Result<String> {
        Ok(match self.db.get_row("memories", id)? {
            Some(Row { values, .. }) => match values.get(1) {
                Some(Value::Text(t)) => t.clone(),
                _ => String::new(),
            },
            None => String::new(),
        })
    }
}

fn main() -> quickdb::Result<()> {
    let dir: PathBuf =
        std::env::temp_dir().join(format!("qdb-agent-memory-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    {
        let mut mem = Memory::open(&dir)?;
        let goal = mem.remember(
            "goal",
            "Ship the offline note-taking app for field teams by March",
            None,
        )?;
        let d1 = mem.remember(
            "decision",
            "Store notes locally in one embedded database; sync later over HTTPS",
            Some(("serves", goal)),
        )?;
        mem.remember(
            "fact",
            "Field tablets have 4 GB of RAM and often no network",
            Some(("constrains", d1)),
        )?;
        mem.remember(
            "decision",
            "Use full-text search so notes can be found by exact words",
            Some(("follows", d1)),
        )?;
        mem.remember(
            "fact",
            "ملاحظات الفريق الميداني تُكتب بالعربية والإنجليزية",
            Some(("about", goal)),
        )?;
        mem.db.flush()?;
    }

    // A new session: reopen and recall.
    let mem = Memory::open(&dir)?;
    for question in ["how do we store notes without network", "الفريق الميداني"] {
        println!("Q: {question}");
        let hits = mem.recall(question, 3)?;
        for (id, score) in &hits {
            println!("  {score:.4}  #{id}  {}", mem.text(*id)?);
        }
        if let Some((best, _)) = hits.first() {
            for r in mem.db.k_hop(*best, None, Direction::Out, 2, 10)? {
                println!("    -> #{} (depth {})  {}", r.id, r.depth, mem.text(r.id)?);
            }
        }
    }

    // Plain filters still work on the same rows.
    let decisions = mem.db.select(
        "memories",
        &Filter::Eq {
            column: "kind".into(),
            value: Value::Text("decision".into()),
        },
        10,
    )?;
    println!("decisions on record: {}", decisions.len());
    assert_eq!(decisions.len(), 2);

    drop(mem);
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
