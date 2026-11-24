//! Full-text search support: index definitions, Arabic-aware text folding, safe query
//! building and Reciprocal Rank Fusion.

use quickdb_api::{Hit, NodeId};
use std::collections::{HashMap, HashSet};

/// The `k` constant of Reciprocal Rank Fusion: a document at 1-based rank `r` in one list
/// contributes `1 / (RRF_K + r)` to its fused score.
pub const RRF_K: f64 = 60.0;

/// Most distinct terms a query keeps; later terms are ignored.
pub const MAX_QUERY_TERMS: usize = 64;

/// Upper bound on the `ef_search` share of the candidate depth taken from each list by
/// `hybrid_search`.
pub const MAX_FUSION_DEPTH: usize = 1024;

/// A full-text index over one or more `Text` columns of a table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextIndexDef {
    /// Index name; must satisfy [`quickdb_api::valid_name`].
    pub name: String,
    /// Table whose rows are indexed.
    pub table: String,
    /// `Text` columns of that table, in index order.
    pub columns: Vec<String>,
}

/// Folds Arabic text for matching: drops harakat, Quranic marks and tatweel, maps alef
/// forms to bare alef, alef maqsura to yeh and teh marbuta to heh. Other characters are
/// kept unchanged.
pub fn fold(text: &str) -> String {
    text.chars()
        .filter_map(|c| match c {
            '\u{0610}'..='\u{061A}'
            | '\u{064B}'..='\u{065F}'
            | '\u{0670}'
            | '\u{06D6}'..='\u{06DC}'
            | '\u{06DF}'..='\u{06E8}'
            | '\u{06EA}'..='\u{06ED}'
            | '\u{0640}' => None,
            '\u{0622}' | '\u{0623}' | '\u{0625}' | '\u{0671}' => Some('\u{0627}'),
            '\u{0649}' => Some('\u{064A}'),
            '\u{0629}' => Some('\u{0647}'),
            c => Some(c),
        })
        .collect()
}

/// All-terms form of [`match_terms`], kept for the unit tests.
#[cfg(test)]
pub(crate) fn match_expression(query: &str) -> Option<String> {
    match_terms(query, false)
}

/// Builds an FTS5 expression from free text: folded, split on every character that is not
/// alphanumeric, each distinct term quoted, terms combined with `AND`, or with `OR` when `any`
/// is true. `None` when the text has no terms.
pub(crate) fn match_terms(query: &str, any: bool) -> Option<String> {
    let folded = fold(query).to_lowercase();
    let mut seen = HashSet::new();
    let mut terms = Vec::new();
    for word in folded.split(|c: char| !c.is_alphanumeric()) {
        if !word.is_empty() && seen.insert(word) {
            terms.push(format!("\"{}\"", word.replace('"', "\"\"")));
            if terms.len() == MAX_QUERY_TERMS {
                break;
            }
        }
    }
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(if any { " OR " } else { " AND " }))
    }
}

/// Name of the FTS5 table behind a text index. The length prefix keeps one index's
/// table from colliding with another index's FTS5 shadow tables.
pub(crate) fn table_name(index: &str) -> String {
    format!("fts{}_{index}", index.len())
}

/// Fuses a text ranking and a vector ranking with Reciprocal Rank Fusion; best first,
/// ties broken by ascending id.
pub(crate) fn fuse(text: &[(NodeId, f64)], vector: &[Hit], k: usize) -> Vec<(NodeId, f64)> {
    let mut scores: HashMap<NodeId, f64> = HashMap::new();
    let ranked = text
        .iter()
        .map(|&(id, _)| id)
        .enumerate()
        .chain(vector.iter().map(|h| h.id).enumerate());
    for (rank, id) in ranked {
        *scores.entry(id).or_insert(0.0) += 1.0 / (RRF_K + rank as f64 + 1.0);
    }
    let mut fused: Vec<_> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    fused.truncate(k);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arabic_folding() {
        assert_eq!(fold("كَتَبَ"), "كتب");
        assert_eq!(fold("أحمد إلى آخر"), "احمد الي اخر");
        assert_eq!(fold("مدرسة"), "مدرسه");
        assert_eq!(fold("كـــتاب"), "كتاب");
        assert_eq!(fold("Café"), "Café");
    }

    #[test]
    fn any_mode_joins_with_or() {
        assert_eq!(
            match_terms("notes, network!", true),
            Some("\"notes\" OR \"network\"".into())
        );
        assert_eq!(match_terms(" ,. ", true), None);
    }

    #[test]
    fn query_terms_are_quoted_and_joined_with_and() {
        assert_eq!(
            match_expression("كُتُب  الطالبِ"),
            Some("\"كتب\" AND \"الطالب\"".into())
        );
        assert_eq!(match_expression("Hello, hello!"), Some("\"hello\"".into()));
        assert_eq!(
            match_expression("NEAR(a\" b*) -c d:e"),
            Some("\"near\" AND \"a\" AND \"b\" AND \"c\" AND \"d\" AND \"e\"".into())
        );
        assert_eq!(match_expression(" ,;\"*()"), None);
        assert_eq!(match_expression(""), None);
        let long: String = (0..1000).map(|i| format!("w{i} ")).collect();
        let expr = match_expression(&long).unwrap();
        assert_eq!(expr.matches(" AND ").count(), MAX_QUERY_TERMS - 1);
    }

    #[test]
    fn table_names_do_not_collide_with_shadow_tables() {
        assert_eq!(table_name("a"), "fts1_a");
        assert_ne!(table_name("a_data"), format!("{}_data", table_name("a")));
    }

    #[test]
    fn fusion_follows_reciprocal_ranks() {
        let text = [(1, 9.0), (2, 5.0)];
        let vector = [
            Hit {
                id: 3,
                distance: 0.1,
            },
            Hit {
                id: 2,
                distance: 0.2,
            },
        ];
        let fused = fuse(&text, &vector, 10);
        let ids: Vec<_> = fused.iter().map(|h| h.0).collect();
        assert_eq!(ids, [2, 1, 3]);
        assert_eq!(fused[0].1, 2.0 / 62.0);
        assert_eq!(fused[1].1, fused[2].1);
        assert_eq!(fuse(&text, &vector, 1).len(), 1);
    }
}
