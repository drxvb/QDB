//! Runs the README quickstart (the example file itself) and checks every result.

#[allow(dead_code)]
#[path = "../examples/readme_quickstart.rs"]
mod quickstart;

use quickdb_api::Reached;

#[test]
fn quickstart_results_are_exact() {
    let dir = tempfile::tempdir().expect("temp dir");
    let r = quickstart::run(&dir.path().join("store")).expect("quickstart runs");

    // Distances to [1,0,0,0] (L2): 1 -> 0, 2 -> 0.02, 6 -> 0.09, then 4, 3, 5 far away.
    assert_eq!(r.nearest, vec![1, 2, 6]);
    // Documents from 2024 on are 3, 4, 5, 6; the closest two are 6 and 4.
    assert_eq!(r.nearest_recent, vec![6, 4]);
    assert_eq!(
        r.two_hops,
        vec![
            Reached { id: 2, depth: 1 },
            Reached { id: 3, depth: 2 },
            Reached { id: 4, depth: 2 }
        ]
    );
    assert_eq!(r.rows_after_reopen, 6);
    assert_eq!(r.vectors_after_reopen, 6);
    assert_eq!(r.edges_after_reopen, 5);
    assert_eq!(r.title_of_3_after_reopen.as_deref(), Some("indexes"));
    assert_eq!(r.nearest_after_reopen, r.nearest);
}
