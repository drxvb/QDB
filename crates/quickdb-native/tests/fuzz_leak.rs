//! Fuzz finding (target `store_api`, LeakSanitizer): every parallel insert batch leaked the
//! adjacency lists it saved for rollback. The batch guard was disarmed with `mem::forget`,
//! which also skipped freeing its `saved` buffer. Input: tests/fixtures/fuzz/store_api-leak-batch-guard.
//!
//! This file holds one test, alone in its own binary, so the counting allocator sees no other
//! test's allocations.
use quickdb_api::{Metric, OpenOptions, Store, VectorIndexDef};
use quickdb_native::NativeStore;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

struct Counting;
static LIVE: AtomicIsize = AtomicIsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.fetch_add(
            new_size as isize - layout.size() as isize,
            Ordering::Relaxed,
        );
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Builds an index past the sequential seed (256 vectors), so it inserts in parallel
/// batches, then reopens it without a snapshot (rebuild path) and drops everything.
fn build_and_drop() {
    let dir = tempfile::tempdir().expect("temp dir");
    {
        let mut s = NativeStore::open(dir.path(), &OpenOptions::default()).expect("open");
        s.define_vector_index(&VectorIndexDef {
            name: "v".into(),
            dim: 8,
            metric: Metric::L2,
            m: 16,
            ef_construction: 64,
        })
        .expect("index");
        let vectors: Vec<Vec<f32>> = (0..2_000u32)
            .map(|i| (0..8).map(|d| ((i * 31 + d * 17) % 97) as f32).collect())
            .collect();
        let items: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u64, v.as_slice()))
            .collect();
        s.upsert_vectors("v", &items).expect("vectors");
    }
    // No flush, so the next open rebuilds the index from SQLite through the same batches.
    let s = NativeStore::open(dir.path(), &OpenOptions::default()).expect("reopen");
    assert_eq!(s.index_rebuilds(), 1);
    drop(s);
}

#[test]
fn parallel_insert_batches_free_their_rollback_copies() {
    // The first round pays one-time allocations (SQLite and std statics).
    build_and_drop();
    let before = LIVE.load(Ordering::Relaxed);
    build_and_drop();
    let after = LIVE.load(Ordering::Relaxed);
    let retained = after - before;
    // Before the fix about 2 x 27 batches leaked their saved lists (several MB here).
    assert!(
        retained < 256 * 1024,
        "{retained} bytes still allocated after the store was dropped"
    );
}
