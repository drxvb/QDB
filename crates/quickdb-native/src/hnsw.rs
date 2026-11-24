// This graph is also included directly by the frozen reference example.
#[path = "reach.rs"]
pub(crate) mod reach;
#[allow(dead_code)]
#[path = "snapshot.rs"]
pub(crate) mod snapshot;

use quickdb_api::{Hit, Metric, NodeId};

use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

trait Visit {
    fn insert(&mut self, node: usize) -> bool;
}
impl Visit for Vec<bool> {
    fn insert(&mut self, node: usize) -> bool {
        if self[node] {
            false
        } else {
            self[node] = true;
            true
        }
    }
}
struct VisitScratch {
    seen: Vec<bool>,
    touched: Vec<usize>,
}
impl VisitScratch {
    fn reset(&mut self, nodes: usize) {
        for node in self.touched.drain(..) {
            self.seen[node] = false;
        }
        self.seen.resize(nodes, false);
    }
}
impl Visit for VisitScratch {
    fn insert(&mut self, node: usize) -> bool {
        if self.seen[node] {
            return false;
        }
        self.seen[node] = true;
        self.touched.push(node);
        true
    }
}
thread_local! {
    static VISIT_SCRATCH: RefCell<VisitScratch> = const {
        RefCell::new(VisitScratch { seen: Vec::new(), touched: Vec::new() })
    };
}
fn with_visited<T>(nodes: usize, f: impl FnOnce(&mut VisitScratch) -> T) -> T {
    VISIT_SCRATCH.with(|cell| {
        if let Ok(mut scratch) = cell.try_borrow_mut() {
            scratch.reset(nodes);
            f(&mut scratch)
        } else {
            let mut scratch = VisitScratch {
                seen: vec![false; nodes],
                touched: Vec::new(),
            };
            f(&mut scratch)
        }
    })
}

#[cfg(target_arch = "x86_64")]
fn use_avx2() -> bool {
    std::is_x86_feature_detected!("avx2") && std::env::var_os("QUICKDB_BUILD_SCALAR").is_none()
}

#[cfg(target_arch = "x86_64")]
fn use_fma() -> bool {
    std::env::var("QUICKDB_BUILD_FMA").as_deref() == Ok("1")
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
}

fn build_margin_pct() -> Option<usize> {
    std::env::var("QUICKDB_BUILD_MARGIN_PCT")
        .ok()
        .and_then(|value| value.parse().ok())
}

fn construction_breadth(ef: usize, margin_pct: Option<usize>) -> usize {
    // Absence (including invalid input) preserves the historical 12.5%, not 12%.
    let margin = margin_pct.map_or(ef / 8, |pct| {
        ((ef as u128 * pct as u128 / 100).min(usize::MAX as u128)) as usize
    });
    ef.saturating_add(margin)
}

/// Hints issued per float vector. One hint per two cache lines for the whole vector made
/// search on 1,024 dimensions 25% slower; caps from 0 to 8 measured alike there, and 4 is
/// what a 128-dimension vector used when prefetch was measured as a gain.
/// The variable exists for measurements.
fn prefetch_hints() -> usize {
    std::env::var("QUICKDB_PREFETCH_HINTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
}

fn small_integer(value: f32) -> Option<u8> {
    if (0.0..=255.0).contains(&value) && value.fract() == 0.0 {
        Some(value as u8)
    } else {
        None
    }
}

fn pack_data(dim: usize, cosine: bool, data: &[f32]) -> Option<Vec<u8>> {
    #[cfg(target_arch = "x86_64")]
    {
        if dim == 0
            || dim > 256
            || cosine
            || !use_avx2()
            || std::env::var_os("QUICKDB_BUILD_NO_PACKED").is_some()
        {
            return None;
        }
        data.iter().copied().map(small_integer).collect()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (dim, cosine, data);
        None
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn distance_packed(a: &[u8], b: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let aligned = a.len().min(b.len()) / 16 * 16;
    let mut sums = _mm256_setzero_si256();
    for offset in (0..aligned).step_by(16) {
        let aa = _mm256_cvtepu8_epi16(_mm_loadu_si128(a.as_ptr().add(offset).cast()));
        let bb = _mm256_cvtepu8_epi16(_mm_loadu_si128(b.as_ptr().add(offset).cast()));
        let d = _mm256_sub_epi16(aa, bb);
        sums = _mm256_add_epi32(sums, _mm256_madd_epi16(d, d));
    }
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sums);
    let mut total: u32 = lanes.iter().map(|&value| value as u32).sum();
    for (&a, &b) in a[aligned..].iter().zip(&b[aligned..]) {
        let d = i32::from(a) - i32::from(b);
        total += (d * d) as u32;
    }
    total as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn distance_avx2(a: &[f32], b: &[f32], cosine: bool) -> f32 {
    use std::arch::x86_64::*;
    let chunks = a.len().min(b.len()) / 8;
    let mut sums = _mm256_setzero_ps();
    if cosine {
        for i in 0..chunks {
            let aa = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let bb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            sums = _mm256_add_ps(sums, _mm256_mul_ps(aa, bb));
        }
    } else {
        for i in 0..chunks {
            let aa = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let bb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let d = _mm256_sub_ps(aa, bb);
            sums = _mm256_add_ps(sums, _mm256_mul_ps(d, d));
        }
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), sums);
    let mut sum: f32 = lanes.iter().sum();
    for (a, b) in a[chunks * 8..].iter().zip(&b[chunks * 8..]) {
        sum += if cosine {
            a * b
        } else {
            let d = a - b;
            d * d
        };
    }
    if cosine {
        1.0 - sum
    } else {
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn distance_avx2_fma(a: &[f32], b: &[f32], cosine: bool) -> f32 {
    use std::arch::x86_64::*;
    let chunks = a.len().min(b.len()) / 8;
    let mut sums = _mm256_setzero_ps();
    if cosine {
        for i in 0..chunks {
            let aa = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let bb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            sums = _mm256_fmadd_ps(aa, bb, sums);
        }
    } else {
        for i in 0..chunks {
            let aa = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let bb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let d = _mm256_sub_ps(aa, bb);
            sums = _mm256_fmadd_ps(d, d, sums);
        }
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), sums);
    let mut sum: f32 = lanes.iter().sum();
    for (a, b) in a[chunks * 8..].iter().zip(&b[chunks * 8..]) {
        sum = if cosine {
            a.mul_add(*b, sum)
        } else {
            let d = a - b;
            d.mul_add(d, sum)
        };
    }
    if cosine {
        1.0 - sum
    } else {
        sum
    }
}

#[derive(Default)]
struct BuildProfile([AtomicU64; 7]);
#[cfg(test)]
thread_local! {
    static PANIC_IN_PARALLEL_CALL: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}
impl BuildProfile {
    fn measure<const ENABLED: bool, T>(&self, stage: usize, f: impl FnOnce() -> T) -> T {
        if ENABLED {
            let start = Instant::now();
            let value = f();
            self.0[stage].fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
            value
        } else {
            f()
        }
    }
}

fn parallel_map<const PROFILE: bool, T: Send>(
    count: usize,
    workers: usize,
    profile: &BuildProfile,
    f: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    #[cfg(test)]
    let panic_here = count > 0
        && PANIC_IN_PARALLEL_CALL.with(|cell| {
            let n = cell.get();
            cell.set(n.saturating_sub(1));
            n == 1
        });
    if workers <= 1 || count <= 1 {
        return (0..count)
            .map(|i| {
                #[cfg(test)]
                if panic_here && i == count / 2 {
                    panic!("injected build worker panic");
                }
                f(i)
            })
            .collect();
    }
    std::thread::scope(|scope| {
        let f = &f;
        let handles: Vec<_> = profile.measure::<PROFILE, _>(5, || {
            (0..count)
                .collect::<Vec<_>>()
                .chunks(count.div_ceil(workers))
                .map(|chunk| {
                    let start = chunk[0];
                    let end = start + chunk.len();
                    scope.spawn(move || {
                        let before = if PROFILE && option_env!("QUICKDB_DISTANCE_COUNTER").is_some()
                        {
                            quickdb_distance_count(0)
                        } else {
                            0
                        };
                        let values = (start..end)
                            .map(|i| {
                                #[cfg(test)]
                                if panic_here && i == count / 2 {
                                    panic!("injected build worker panic");
                                }
                                f(i)
                            })
                            .collect::<Vec<_>>();
                        if PROFILE && option_env!("QUICKDB_DISTANCE_COUNTER").is_some() {
                            profile.0[6].fetch_add(
                                quickdb_distance_count(0).wrapping_sub(before),
                                AtomicOrdering::Relaxed,
                            );
                        }
                        values
                    })
                })
                .collect()
        });
        handles
            .into_iter()
            .flat_map(|handle| match handle.join() {
                Ok(values) => values,
                Err(panic) => std::panic::resume_unwind(panic),
            })
            .collect()
    })
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    distance: f32,
    node: usize,
}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance.to_bits() == other.distance.to_bits() && self.node == other.node
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.node.cmp(&other.node))
    }
}

#[derive(Clone, Copy, Debug)]
struct ResultCandidate {
    distance: f32,
    id: NodeId,
    node: usize,
}
impl PartialEq for ResultCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance.to_bits() == other.distance.to_bits()
            && self.id == other.id
            && self.node == other.node
    }
}
impl Eq for ResultCandidate {}
impl PartialOrd for ResultCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ResultCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.id.cmp(&other.id))
            .then(self.node.cmp(&other.node))
    }
}

struct BatchGuard<'a> {
    h: &'a mut Hnsw,
    base: usize,
    was_packed: bool,
    saved: Vec<(usize, usize, Vec<usize>)>,
    // Set once the batch is published; dropping the guard then only frees `saved`.
    committed: bool,
}
impl Drop for BatchGuard<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        for (node, level, links) in self.saved.drain(..) {
            self.h.links[node][level] = links;
        }
        self.h.ids.truncate(self.base);
        self.h.links.truncate(self.base);
        self.h.data.truncate(self.base * self.h.dim);
        if let Some(packed) = &mut self.h.packed {
            packed.truncate(self.base * self.h.dim);
        } else if self.was_packed {
            self.h.packed = pack_data(self.h.dim, self.h.cosine, &self.h.data);
        }
    }
}

/// Snapshot includes the graph, not merely vectors. SQL is the recovery authority.
#[derive(Clone)]
pub(crate) struct Hnsw {
    pub revision: u64,
    dim: usize,
    cosine: bool,
    m: usize,
    ef: usize,
    data: Vec<f32>,
    packed: Option<Vec<u8>>,
    ids: Vec<NodeId>,
    links: Vec<Vec<Vec<usize>>>,
    active: HashMap<NodeId, usize>,
    entry: Option<usize>,
    top: usize,
    reuse_visited: bool,
    prefetch: bool,
    // Read only by the x86_64 `_mm_prefetch` loop; other targets keep it so the layout is the same everywhere.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    prefetch_hints: usize,
    build_margin_pct: Option<usize>,
    /// Search-only layer-0 out-links added by `reach`; never saved, never pruned.
    bridges: Vec<Vec<usize>>,
    #[cfg(target_arch = "x86_64")]
    fma: bool,
    #[cfg(target_arch = "x86_64")]
    avx2: bool,
}
impl Hnsw {
    pub(crate) fn encode_snapshot(&self, out: &mut impl std::io::Write) -> std::io::Result<()> {
        let parts = snapshot::GraphParts {
            revision: self.revision,
            dim: self.dim,
            cosine: self.cosine,
            m: self.m,
            ef: self.ef,
            data: &self.data,
            ids: &self.ids,
            links: &self.links,
            entry: self.entry,
            top: self.top,
        };
        snapshot::encode_graph(out, &parts, self.active.iter().map(|(&id, &n)| (id, n)))
    }

    pub(crate) fn from_snapshot(graph: snapshot::Graph) -> std::io::Result<Self> {
        let snapshot::Graph {
            revision,
            dim,
            cosine,
            m,
            ef,
            data,
            ids,
            links,
            active,
            entry,
            top,
        } = graph;
        let mut map = HashMap::new();
        map.try_reserve(active.len()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "snapshot map allocation failed",
            )
        })?;
        for (id, node) in active {
            if map.insert(id, node).is_some() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "duplicate snapshot id",
                ));
            }
        }
        let packed = pack_data(dim, cosine, &data);
        let mut h = Self {
            revision,
            dim,
            cosine,
            m,
            ef,
            data,
            packed,
            ids,
            links,
            active: map,
            entry,
            top,
            reuse_visited: std::env::var_os("QUICKDB_BUILD_FRESH_VISITED").is_none(),
            prefetch: std::env::var_os("QUICKDB_BUILD_NO_PREFETCH").is_none(),
            prefetch_hints: prefetch_hints(),
            build_margin_pct: build_margin_pct(),
            bridges: Vec::new(),
            #[cfg(target_arch = "x86_64")]
            fma: use_fma(),
            #[cfg(target_arch = "x86_64")]
            avx2: use_avx2(),
        };
        // Bridges are not in the snapshot, so a loaded graph is repaired again.
        h.reach_after_build();
        Ok(h)
    }
}
impl Hnsw {
    pub fn new(dim: usize, metric: Metric, m: usize, ef: usize) -> Self {
        Self {
            revision: 0,
            dim,
            cosine: metric == Metric::Cosine,
            m: m.max(2),
            ef: ef.max(m).max(16),
            data: Vec::new(),
            packed: pack_data(dim, metric == Metric::Cosine, &[]),
            ids: Vec::new(),
            links: Vec::new(),
            active: HashMap::new(),
            entry: None,
            top: 0,
            reuse_visited: std::env::var_os("QUICKDB_BUILD_FRESH_VISITED").is_none(),
            prefetch: std::env::var_os("QUICKDB_BUILD_NO_PREFETCH").is_none(),
            prefetch_hints: prefetch_hints(),
            build_margin_pct: build_margin_pct(),
            bridges: Vec::new(),
            #[cfg(target_arch = "x86_64")]
            fma: use_fma(),
            #[cfg(target_arch = "x86_64")]
            avx2: use_avx2(),
        }
    }
    pub fn len(&self) -> usize {
        self.active.len()
    }
    pub fn needs_compaction(&self) -> bool {
        self.ids.len().saturating_sub(self.active.len()) > self.ids.len() / 5
    }
    #[cfg(test)]
    pub fn node_count(&self) -> usize {
        self.ids.len()
    }
    pub fn valid(&self, dim: usize, metric: Metric, m: usize, ef: usize) -> bool {
        self.dim == dim
            && self.cosine == (metric == Metric::Cosine)
            && self.m == m.max(2)
            && self.ef == ef.max(m).max(16)
            && self.data.len() == self.ids.len().saturating_mul(dim)
            && self.links.len() == self.ids.len()
            && self.data.iter().all(|v| v.is_finite())
            && self.ids.iter().all(|id| *id <= i64::MAX as u64)
            && self
                .active
                .iter()
                .all(|(id, n)| self.ids.get(*n) == Some(id))
            && self.entry.is_none() == self.ids.is_empty()
            && self.top <= 24
            && self.entry.is_none_or(|n| {
                n < self.ids.len() && Some(self.links[n].len()) == self.top.checked_add(1)
            })
            && self.links.iter().enumerate().all(|(node, layers)| {
                layers.len() == self.level(node) + 1
                    && layers.len() <= self.top + 1
                    && layers.iter().enumerate().all(|(level, ns)| {
                        ns.iter()
                            .all(|n| self.links.get(*n).is_some_and(|ls| ls.len() > level))
                    })
            })
    }
    fn vector(&self, node: usize) -> &[f32] {
        &self.data[node * self.dim..(node + 1) * self.dim]
    }
    fn append_vector(&mut self, vector: &[f32]) {
        if self.packed.is_some() && vector.iter().all(|&value| small_integer(value).is_some()) {
            if let Some(packed) = &mut self.packed {
                packed.extend(vector.iter().map(|&value| value as u8));
            }
        } else {
            self.packed = None;
        }
        self.data.extend_from_slice(vector);
    }
    fn packed_slice(&self, vector: &[f32]) -> Option<&[u8]> {
        let packed = self.packed.as_ref()?;
        if vector.len() != self.dim {
            return None;
        }
        let bytes = (vector.as_ptr() as usize).checked_sub(self.data.as_ptr() as usize)?;
        if !bytes.is_multiple_of(std::mem::size_of::<f32>()) {
            return None;
        }
        let start = bytes / std::mem::size_of::<f32>();
        packed.get(start..start.checked_add(self.dim)?)
    }
    fn prepare(&self, v: &[f32]) -> Vec<f32> {
        let mut v = v.to_vec();
        if self.cosine {
            let norm = v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
            if norm > 0.0 {
                for x in &mut v {
                    *x = (f64::from(*x) / norm) as f32;
                }
            }
        }
        v
    }
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if option_env!("QUICKDB_DISTANCE_COUNTER").is_some() {
            quickdb_distance_count(2);
        }
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            if let Some(a) = self.packed_slice(a) {
                if let Some(b) = self.packed_slice(b) {
                    // The paired 200k probe fell from 74.89s to 49.07s.
                    // Eligible L2 vectors have <=256 integer components in
                    // [0,255]. Every product and partial sum is an exact f32
                    // integer (total <=16,646,400 <2^24), so integer grouping
                    // preserves the old bits. AVX2 is checked and 16-byte loads
                    // are bounded by both slices. The cache is never persisted.
                    return unsafe { distance_packed(a, b) };
                }
            }
            if self.fma {
                // Opt-in experiment: FMA saves a multiply/add instruction in each
                // chunk. AVX2/FMA were checked at creation/deserialization and
                // every unaligned load is bounded by both slices. Queries use
                // the same dispatch, with no kernel metadata in the snapshot.
                return unsafe { distance_avx2_fma(a, b, self.cosine) };
            }
            // The paired 200k SIFT probe measured 171.67s scalar versus 142.29s
            // AVX2. AVX2 was checked on creation/deserialization. Each unaligned
            // load is bounded by the shorter slice. The kernel retains the
            // original eight accumulators and does not fuse multiply/add.
            return unsafe { distance_avx2(a, b, self.cosine) };
        }
        self.distance_scalar(a, b)
    }
    fn distance_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        // Independent accumulators let LLVM vectorize without fast-math reordering.
        let chunks = a.len().min(b.len()) / 8;
        let mut sums = [0.0f32; 8];
        let aa = a.chunks_exact(8);
        let bb = b.chunks_exact(8);
        for (a, b) in aa.zip(bb) {
            for j in 0..8 {
                sums[j] += if self.cosine {
                    a[j] * b[j]
                } else {
                    let d = a[j] - b[j];
                    d * d
                };
            }
        }
        let mut sum: f32 = sums.iter().sum();
        for (a, b) in a[chunks * 8..].iter().zip(&b[chunks * 8..]) {
            sum += if self.cosine {
                a * b
            } else {
                let d = a - b;
                d * d
            };
        }
        if self.cosine {
            1.0 - sum
        } else {
            sum
        }
    }
    fn candidate(&self, q: &[f32], node: usize) -> Candidate {
        Candidate {
            node,
            distance: self.distance(q, self.vector(node)),
        }
    }
    fn result_candidate(&self, q: &[f32], node: usize) -> ResultCandidate {
        ResultCandidate {
            distance: self.distance(q, self.vector(node)),
            id: self.ids[node],
            node,
        }
    }
    fn prefetch_vector(&self, node: usize, packed_query: bool) {
        #[cfg(target_arch = "x86_64")]
        {
            if packed_query {
                if let Some(packed) = &self.packed {
                    let vector = &packed[node * self.dim..(node + 1) * self.dim];
                    for offset in (0..vector.len()).step_by(64) {
                        // The address is inside a live byte-cache slice. SSE
                        // prefetch is available on every x86_64 CPU.
                        unsafe {
                            std::arch::x86_64::_mm_prefetch(
                                vector.as_ptr().add(offset).cast(),
                                std::arch::x86_64::_MM_HINT_T0,
                            );
                        }
                    }
                    return;
                }
            }
            let vector = self.vector(node);
            for offset in (0..vector.len()).step_by(32).take(self.prefetch_hints) {
                // The paired 200k probe fell from 129.22s to 117.11s. x86_64
                // guarantees SSE prefetch support. Each hinted address is within
                // this live slice. This does not read or alter a float.
                unsafe {
                    std::arch::x86_64::_mm_prefetch(
                        vector.as_ptr().add(offset).cast(),
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (node, packed_query);
    }
    fn greedy(&self, q: &[f32], mut best: Candidate, level: usize) -> Candidate {
        loop {
            let old = best.node;
            for &n in &self.links[old][level] {
                let next = self.candidate(q, n);
                if next < best {
                    best = next;
                }
            }
            if best.node == old {
                return best;
            }
        }
    }
    fn layer(&self, q: &[f32], entry: usize, ef: usize, level: usize) -> Vec<ResultCandidate> {
        let mut visited = vec![false; self.ids.len()];
        let packed_query = self.prefetch && self.packed_slice(q).is_some();
        let first = self.result_candidate(q, entry);
        let mut frontier = BinaryHeap::from([Reverse(first)]);
        let mut best = BinaryHeap::from([first]);
        visited[entry] = true;
        while let Some(Reverse(c)) = frontier.pop() {
            if best.len() >= ef && best.peek().is_some_and(|worst| c > *worst) {
                break;
            }
            let neighbors = &self.links[c.node][level];
            if self.prefetch {
                for &n in neighbors.iter().take(4) {
                    self.prefetch_vector(n, packed_query);
                }
            }
            for (offset, &n) in neighbors.iter().enumerate() {
                if self.prefetch {
                    if let Some(&future) = neighbors.get(offset + 4) {
                        self.prefetch_vector(future, packed_query);
                    }
                }
                if visited[n] {
                    continue;
                }
                visited[n] = true;
                let next = self.result_candidate(q, n);
                if best.len() < ef || best.peek().is_some_and(|worst| next < *worst) {
                    frontier.push(Reverse(next));
                    best.push(next);
                    if best.len() > ef {
                        best.pop();
                    }
                }
            }
            if level == 0 {
                for &n in self.bridge_links(c.node) {
                    if visited[n] {
                        continue;
                    }
                    visited[n] = true;
                    let next = self.result_candidate(q, n);
                    if best.len() < ef || best.peek().is_some_and(|worst| next < *worst) {
                        frontier.push(Reverse(next));
                        best.push(next);
                        if best.len() > ef {
                            best.pop();
                        }
                    }
                }
            }
        }
        best.into_sorted_vec()
    }
    fn layer_build(&self, q: &[f32], entry: usize, ef: usize, level: usize) -> Vec<Candidate> {
        if self.reuse_visited {
            with_visited(self.ids.len(), |visited| {
                self.layer_walk(q, entry, ef, level, visited)
            })
        } else {
            let mut visited = vec![false; self.ids.len()];
            self.layer_walk(q, entry, ef, level, &mut visited)
        }
    }
    fn layer_walk(
        &self,
        q: &[f32],
        entry: usize,
        ef: usize,
        level: usize,
        visited: &mut impl Visit,
    ) -> Vec<Candidate> {
        let packed_query = self.prefetch && self.packed_slice(q).is_some();
        let first = self.candidate(q, entry);
        let mut frontier = BinaryHeap::from([Reverse(first)]);
        let mut best = BinaryHeap::from([first]);
        let float_prefetch = self.prefetch
            && self.dim > 256
            && std::env::var("QUICKDB_BUILD_FLOAT_OFF").as_deref() == Ok("0");
        let mut fresh = Vec::new();
        visited.insert(entry);
        while let Some(Reverse(c)) = frontier.pop() {
            if best.len() >= ef && best.peek().is_some_and(|worst| c > *worst) {
                break;
            }
            let neighbors = &self.links[c.node][level];
            if float_prefetch {
                // Mark in link order before issuing hints. Only first visits are
                // scored, in the same order as the original walk. Repeated links
                // no longer consume prefetch bandwidth or the four-slot lookahead.
                fresh.clear();
                fresh.extend(neighbors.iter().copied().filter(|&n| visited.insert(n)));
                for &n in fresh.iter().take(4) {
                    self.prefetch_vector(n, packed_query);
                }
                for (offset, &n) in fresh.iter().enumerate() {
                    if let Some(&future) = fresh.get(offset + 4) {
                        self.prefetch_vector(future, packed_query);
                    }
                    let next = self.candidate(q, n);
                    if best.len() < ef || best.peek().is_some_and(|worst| next < *worst) {
                        frontier.push(Reverse(next));
                        best.push(next);
                        if best.len() > ef {
                            best.pop();
                        }
                    }
                }
                continue;
            }
            if self.prefetch {
                for &n in neighbors.iter().take(4) {
                    self.prefetch_vector(n, packed_query);
                }
            }
            for (offset, &n) in neighbors.iter().enumerate() {
                if self.prefetch {
                    if let Some(&future) = neighbors.get(offset + 4) {
                        self.prefetch_vector(future, packed_query);
                    }
                }
                if !visited.insert(n) {
                    continue;
                }
                let next = self.candidate(q, n);
                if best.len() < ef || best.peek().is_some_and(|worst| next < *worst) {
                    frontier.push(Reverse(next));
                    best.push(next);
                    if best.len() > ef {
                        best.pop();
                    }
                }
            }
        }
        best.into_sorted_vec()
    }
    fn diverse(&self, candidates: Vec<Candidate>, capacity: usize) -> Vec<usize> {
        let mut chosen = Vec::with_capacity(capacity);
        for c in candidates {
            if chosen
                .iter()
                .all(|&n| self.distance(self.vector(c.node), self.vector(n)) >= c.distance)
            {
                chosen.push(c.node);
                if chosen.len() == capacity {
                    break;
                }
            }
        }
        chosen
    }
    fn level(&self, node: usize) -> usize {
        // SplitMix64 gives deterministic geometric levels independent of id order.
        let mut hash = (node as u64).wrapping_add(0x9e3779b97f4a7c15);
        hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d049bb133111eb);
        hash ^= hash >> 31;
        let mut level = 0;
        while hash.is_multiple_of(self.m as u64) && level < 24 {
            level += 1;
            hash /= self.m as u64;
        }
        level
    }
    pub fn insert(&mut self, id: NodeId, vector: &[f32]) {
        let q = self.prepare(vector);
        let node = self.ids.len();
        let level = self.level(node);
        self.append_vector(&q);
        self.ids.push(id);
        self.links.push(vec![Vec::new(); level + 1]);
        self.active.insert(id, node);
        let Some(entry) = self.entry else {
            self.entry = Some(node);
            self.top = level;
            return;
        };
        let mut nearest = self.candidate(&q, entry);
        for l in ((level + 1)..=self.top).rev() {
            nearest = self.greedy(&q, nearest, l);
        }
        for l in (0..=level.min(self.top)).rev() {
            let found = self.layer_build(&q, nearest.node, self.ef, l);
            if let Some(first) = found.first() {
                nearest = *first;
            }
            let selected = self.diverse(found, self.m);
            self.links[node][l] = selected.clone();
            for n in selected {
                self.links[n][l].push(node);
                let capacity = if l == 0 { 2 * self.m } else { self.m };
                if self.links[n][l].len() > capacity {
                    let mut candidates: Vec<_> = self.links[n][l]
                        .iter()
                        .map(|&x| self.candidate(self.vector(n), x))
                        .collect();
                    candidates.sort_unstable();
                    self.links[n][l] = self.diverse(candidates, capacity);
                }
            }
        }
        if level > self.top {
            self.entry = Some(node);
            self.top = level;
        }
    }
    pub fn insert_many(&mut self, items: &[(NodeId, &[f32])]) {
        // The graph does not depend on the worker count; the variable exists for measurements.
        let workers = std::env::var("QUICKDB_BUILD_WORKERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map_or(1, usize::from)
                    .min(8)
            });
        self.insert_many_with_workers(items, workers);
    }
    fn insert_many_with_workers(&mut self, items: &[(NodeId, &[f32])], workers: usize) {
        if std::env::var_os("QUICKDB_BUILD_PROFILE").is_some() {
            self.insert_many_profiled::<true>(items, workers);
        } else {
            self.insert_many_profiled::<false>(items, workers);
        }
        self.reach_after_build();
    }
    fn insert_many_profiled<const PROFILE: bool>(
        &mut self,
        items: &[(NodeId, &[f32])],
        workers: usize,
    ) {
        let profile = BuildProfile::default();
        let count_before = if PROFILE && option_env!("QUICKDB_DISTANCE_COUNTER").is_some() {
            quickdb_distance_count(0)
        } else {
            0
        };
        let total = PROFILE.then(Instant::now);
        // A fixed seed and fixed batch boundaries make scheduling irrelevant.
        // Search and pruning read immutable graphs. Only this thread publishes links.
        let seed = 256usize.saturating_sub(self.ids.len()).min(items.len());
        for &(id, vector) in &items[..seed] {
            self.insert(id, vector);
        }
        for batch in items[seed..].chunks(64) {
            let base = self.ids.len();
            let Some(entry) = self.entry else { continue };
            let was_packed = self.packed.is_some();
            let mut guard = BatchGuard {
                h: self,
                base,
                was_packed,
                saved: Vec::new(),
                committed: false,
            };
            for &(id, vector) in batch {
                let node = guard.h.ids.len();
                let q = guard.h.prepare(vector);
                guard.h.append_vector(&q);
                guard.h.ids.push(id);
                guard
                    .h
                    .links
                    .push(vec![Vec::new(); guard.h.level(node) + 1]);
            }
            let h: &Hnsw = guard.h;
            let plans = parallel_map::<PROFILE, _>(batch.len(), workers, &profile, |offset| {
                let node = base + offset;
                let q = h.vector(node);
                let level = h.links[node].len() - 1;
                let mut nearest = profile.measure::<PROFILE, _>(0, || {
                    let mut nearest = h.candidate(q, entry);
                    for l in ((level + 1)..=h.top).rev() {
                        nearest = h.greedy(q, nearest, l);
                    }
                    nearest
                });
                let mut layers = vec![Vec::new(); level + 1];
                for l in (0..=level.min(h.top)).rev() {
                    // Frozen batches omit graph changes from earlier peers. A 12.5%
                    // construction margin compensates without changing search ef.
                    let mut found = profile.measure::<PROFILE, _>(0, || {
                        h.layer_build(
                            q,
                            nearest.node,
                            construction_breadth(h.ef, h.build_margin_pct),
                            l,
                        )
                    });
                    if let Some(first) = found.first() {
                        nearest = *first;
                    }
                    // Peers are scored directly, never traversed until publication.
                    profile.measure::<PROFILE, _>(1, || {
                        found.extend(
                            (base..h.ids.len())
                                .filter(|&peer| peer != node && h.links[peer].len() > l)
                                .map(|peer| h.candidate(q, peer)),
                        )
                    });
                    layers[l] = profile.measure::<PROFILE, _>(2, || {
                        found.sort_unstable();
                        h.diverse(found, h.m)
                    });
                }
                layers
            });
            let affected = profile.measure::<PROFILE, _>(3, || {
                for (offset, layers) in plans.into_iter().enumerate() {
                    guard.h.links[base + offset] = layers;
                }
                let mut affected = Vec::new();
                for node in base..guard.h.ids.len() {
                    for level in 0..guard.h.links[node].len() {
                        affected.push((node, level));
                        for &neighbor in &guard.h.links[node][level] {
                            affected.push((neighbor, level));
                        }
                    }
                }
                affected.sort_unstable();
                affected.dedup();
                for &(node, level) in &affected {
                    if node < base {
                        guard
                            .saved
                            .push((node, level, guard.h.links[node][level].clone()));
                    }
                }
                for node in base..guard.h.ids.len() {
                    for level in 0..guard.h.links[node].len() {
                        let selected = guard.h.links[node][level].clone();
                        for neighbor in selected {
                            guard.h.links[neighbor][level].push(node);
                        }
                    }
                }
                for &(node, level) in &affected {
                    guard.h.links[node][level].sort_unstable();
                    guard.h.links[node][level].dedup();
                }
                affected
            });
            let h: &Hnsw = guard.h;
            let pruned = parallel_map::<PROFILE, _>(affected.len(), workers, &profile, |offset| {
                profile.measure::<PROFILE, _>(4, || {
                    let (node, level) = affected[offset];
                    let capacity = if level == 0 { 2 * h.m } else { h.m };
                    if h.links[node][level].len() <= capacity {
                        return None;
                    }
                    let mut candidates: Vec<_> = h.links[node][level]
                        .iter()
                        .map(|&n| h.candidate(h.vector(node), n))
                        .collect();
                    candidates.sort_unstable();
                    Some(h.diverse(candidates, capacity))
                })
            });
            profile.measure::<PROFILE, _>(3, || {
                for ((node, level), neighbors) in affected.into_iter().zip(pruned) {
                    if let Some(neighbors) = neighbors {
                        guard.h.links[node][level] = neighbors;
                    }
                }
                for (offset, &(id, _)) in batch.iter().enumerate() {
                    let node = base + offset;
                    guard.h.active.insert(id, node);
                    let level = guard.h.links[node].len() - 1;
                    if level > guard.h.top {
                        guard.h.entry = Some(node);
                        guard.h.top = level;
                    }
                }
            });
            // Disarm the rollback but still free the saved lists (`mem::forget` leaked them).
            guard.committed = true;
            drop(guard);
        }
        if let Some(total) = total {
            if option_env!("QUICKDB_DISTANCE_COUNTER").is_some() {
                let count = quickdb_distance_count(0).wrapping_sub(count_before)
                    + profile.0[6].load(AtomicOrdering::Relaxed);
                eprintln!(
                    "BUILD_DISTANCE inserts={} count={count} per_insert={:.3}",
                    items.len(),
                    count as f64 / items.len().max(1) as f64
                );
            }
            eprintln!("BUILD_PROFILE graph_ns={} search_ns={} peer_ns={} selection_ns={} publish_ns={} prune_ns={} thread_start_ns={} worker_distances={}", total.elapsed().as_nanos(), profile.0[0].load(AtomicOrdering::Relaxed), profile.0[1].load(AtomicOrdering::Relaxed), profile.0[2].load(AtomicOrdering::Relaxed), profile.0[3].load(AtomicOrdering::Relaxed), profile.0[4].load(AtomicOrdering::Relaxed), profile.0[5].load(AtomicOrdering::Relaxed), profile.0[6].load(AtomicOrdering::Relaxed));
        }
    }
    pub fn exact(
        &self,
        query: &[f32],
        k: usize,
        allowed: Option<&std::collections::HashSet<NodeId>>,
    ) -> Vec<Hit> {
        let q = self.prepare(query);
        let mut hits: Vec<_> = self
            .active
            .iter()
            .filter(|(id, _)| allowed.is_none_or(|a| a.contains(id)))
            .map(|(&id, &n)| Hit {
                id,
                distance: self.distance(&q, self.vector(n)),
            })
            .collect();
        sort_hits(&mut hits, k);
        hits
    }
    /// Internal position of the live vector for `id`.
    pub fn node_of(&self, id: NodeId) -> Option<usize> {
        self.active.get(&id).copied()
    }
    pub fn id_at(&self, node: usize) -> NodeId {
        self.ids[node]
    }
    /// Positions ever allocated, including tombstones; the bound for node masks.
    pub fn capacity(&self) -> usize {
        self.ids.len()
    }
    fn live(&self, node: usize) -> bool {
        self.active.get(&self.ids[node]) == Some(&node)
    }
    /// Exact scoring of the given positions only; tombstoned positions are skipped.
    pub fn exact_nodes(&self, query: &[f32], k: usize, nodes: &[usize]) -> Vec<Hit> {
        let q = self.prepare(query);
        let mut hits: Vec<_> = nodes
            .iter()
            .filter(|&&n| n < self.ids.len() && self.live(n))
            .map(|&n| Hit {
                id: self.ids[n],
                distance: self.distance(&q, self.vector(n)),
            })
            .collect();
        sort_hits(&mut hits, k);
        hits
    }
    /// Layer-0 search where every node is traversed but only live nodes accepted by
    /// `allow` enter the result heap. The walk stops when the nearest unexpanded
    /// candidate is farther than the worst of `ef` accepted results, or after
    /// `budget` distance computations. `allow` receives the positions of one
    /// expansion that could enter the results and marks the accepted ones, so a
    /// costly predicate is asked once per expansion, not once per node. Returns the
    /// hits and whether the walk ended on its own (false: budget exhausted).
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        budget: usize,
        width: usize,
        mut allow: impl FnMut(&[usize], &mut [bool]),
    ) -> (Vec<Hit>, bool) {
        let Some(entry) = self.entry else {
            return (Vec::new(), true);
        };
        if k == 0 {
            return (Vec::new(), true);
        }
        let ef = ef.max(k).min(self.ids.len());
        let q = self.prepare(query);
        let mut best = self.candidate(&q, entry);
        for l in (1..=self.top).rev() {
            best = self.greedy(&q, best, l);
        }
        let first = ResultCandidate {
            distance: best.distance,
            id: self.ids[best.node],
            node: best.node,
        };
        let mut frontier = BinaryHeap::from([Reverse(first)]);
        let mut results: BinaryHeap<ResultCandidate> = BinaryHeap::with_capacity(ef + 1);
        let mut visited = vec![false; self.ids.len()];
        visited[first.node] = true;
        let mut accepted = vec![false; 1];
        if self.live(first.node) {
            allow(&[first.node], &mut accepted);
            if accepted[0] {
                results.push(first);
            }
        }
        let mut spent = 0usize;
        let mut natural = true;
        let mut batch: Vec<ResultCandidate> = Vec::new();
        let mut asked: Vec<usize> = Vec::new();
        // `width` > 1 expands several frontier nodes per predicate call, trading a
        // little extra traversal for fewer calls when the predicate is a query.
        let mut expand: Vec<usize> = Vec::new();
        loop {
            expand.clear();
            while expand.len() < width.max(1) {
                let Some(Reverse(c)) = frontier.pop() else {
                    break;
                };
                if results.len() >= ef && results.peek().is_some_and(|worst| c > *worst) {
                    frontier.clear();
                    break;
                }
                expand.push(c.node);
            }
            if expand.is_empty() {
                break;
            }
            batch.clear();
            for &n in expand
                .iter()
                .flat_map(|&e| self.links[e][0].iter().chain(self.bridge_links(e)))
            {
                if visited[n] {
                    continue;
                }
                if spent >= budget {
                    natural = false;
                    break;
                }
                visited[n] = true;
                spent += 1;
                let next = self.result_candidate(&q, n);
                if results.len() < ef || results.peek().is_some_and(|worst| next < *worst) {
                    frontier.push(Reverse(next));
                    batch.push(next);
                }
            }
            asked.clear();
            asked.extend(batch.iter().map(|b| b.node).filter(|&n| self.live(n)));
            accepted.clear();
            accepted.resize(asked.len(), false);
            if !asked.is_empty() {
                allow(&asked, &mut accepted);
            }
            for (&n, &ok) in asked.iter().zip(&accepted) {
                if ok {
                    if let Some(next) = batch.iter().find(|b| b.node == n) {
                        results.push(*next);
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
            if !natural {
                break;
            }
        }
        let mut hits: Vec<_> = results
            .into_iter()
            .map(|c| Hit {
                id: c.id,
                distance: c.distance,
            })
            .collect();
        sort_hits(&mut hits, k);
        (hits, natural)
    }
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<Hit> {
        if k == 0 {
            return Vec::new();
        }
        if self.len() <= 512 || ef >= self.ids.len() {
            return self.exact(query, k, None);
        }
        let Some(entry) = self.entry else {
            return Vec::new();
        };
        let q = self.prepare(query);
        let mut best = self.candidate(&q, entry);
        for l in (1..=self.top).rev() {
            best = self.greedy(&q, best, l);
        }
        let mut breadth = ef.max(k).max(1).min(self.ids.len());
        loop {
            let mut hits: Vec<_> = self
                .layer(&q, best.node, breadth, 0)
                .into_iter()
                .filter(|c| self.active.get(&self.ids[c.node]) == Some(&c.node))
                .map(|c| Hit {
                    id: self.ids[c.node],
                    distance: c.distance,
                })
                .collect();
            if hits.len() >= k.min(self.len()) || breadth == self.ids.len() {
                sort_hits(&mut hits, k);
                return hits;
            }
            breadth = breadth.saturating_mul(2).min(self.ids.len());
        }
    }
}
pub(crate) fn sort_hits(hits: &mut Vec<Hit>, k: usize) {
    hits.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
    hits.truncate(k);
}

#[cfg(test)]
include!("repair_tests.rs");

#[cfg(test)]
mod build_tests {
    use super::*;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/build_legacy.rs"
    ));

    #[test]
    fn construction_margin_preserves_unset_rounding_and_saturates() {
        for ef in [0, 1, 7, 8, 16, 199, 200, 201, usize::MAX] {
            assert_eq!(construction_breadth(ef, None), ef.saturating_add(ef / 8));
            assert_eq!(construction_breadth(ef, Some(0)), ef);
        }
        assert_eq!(construction_breadth(200, Some(6)), 212);
        assert_eq!(construction_breadth(200, Some(12)), 224);
        assert_eq!(construction_breadth(200, None), 225);
        assert_eq!(
            construction_breadth(usize::MAX, Some(usize::MAX)),
            usize::MAX
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn fma_distances_match_contract_for_chunks_tails_and_zero() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let mut rng = quickdb_api::conformance::Lcg(0x666d61);
        for metric in [Metric::L2, Metric::Cosine] {
            for dim in [1, 7, 8, 9, 16, 31, 32, 127, 128, 129, 960, 1024] {
                let mut h = Hnsw::new(dim, metric, 16, 200);
                h.avx2 = true;
                h.fma = true;
                for _ in 0..20 {
                    let a = h.prepare(&rng.vector(dim));
                    let b = h.prepare(&rng.vector(dim));
                    let actual = h.distance(&a, &b);
                    let expected = h.distance_scalar(&a, &b);
                    assert!((actual - expected).abs() <= 1e-5 * (1.0 + expected.abs()));
                }
                let zero = vec![0.0; dim];
                assert_eq!(h.distance(&zero, &zero), if h.cosine { 1.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn workers_one_two_seven_have_identical_graphs() {
        let mut rng = quickdb_api::conformance::Lcg(739);
        let data: Vec<_> = (0..2048).map(|id| (id, rng.vector(32))).collect();
        let items: Vec<_> = data.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let mut reference = Hnsw::new(32, Metric::L2, 16, 200);
        reference.reuse_visited = false;
        reference.prefetch = false;
        #[cfg(target_arch = "x86_64")]
        {
            reference.avx2 = false;
        }
        reference.insert_many_with_workers(&items, 1);
        for workers in [2, 7] {
            let mut actual = Hnsw::new(32, Metric::L2, 16, 200);
            actual.insert_many_with_workers(&items, workers);
            assert_eq!(reference.data, actual.data);
            assert_eq!(reference.ids, actual.ids);
            assert_eq!(reference.links, actual.links);
            assert_eq!(reference.active, actual.active);
            assert_eq!(reference.entry, actual.entry);
            assert_eq!(reference.top, actual.top);
            for (_, query) in data.iter().step_by(97) {
                assert_eq!(
                    reference.search(query, 10, 64),
                    actual.search(query, 10, 64)
                );
            }
        }
        println!("BUILD_DETERMINISM workers=1,2,7 graph_identical=true");
    }

    #[test]
    fn legacy_snapshot_loads_without_rebuild_and_new_payload_uses_old_schema() {
        use quickdb_api::{OpenOptions, Store, VectorIndexDef};
        let root = std::env::var_os("QUICKDB_DATA")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let dir = tempfile::tempdir_in(root).expect("test directory");
        let mut store =
            crate::NativeStore::open(dir.path(), &OpenOptions::default()).expect("store");
        store
            .define_vector_index(&VectorIndexDef {
                name: "emb".into(),
                dim: 2,
                metric: Metric::L2,
                m: 16,
                ef_construction: 200,
            })
            .expect("index");
        store
            .upsert_vectors("emb", &[(10, &[1.0, 2.0]), (20, &[3.0, 4.0])])
            .expect("vectors");
        store.flush().expect("flush");
        drop(store);
        let snapshot = dir.path().join("emb.hnsw");
        let (header, _) = crate::load_snapshot(&snapshot).expect("snapshot header");
        let mut legacy: Vec<u8> = BUILD_LEGACY_GRAPH_HEX
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let text = std::str::from_utf8(pair).expect("fixture hex");
                u8::from_str_radix(text, 16).expect("fixture byte")
            })
            .collect();
        let decoded = snapshot::decode_graph(legacy.as_slice(), legacy.len() as u64)
            .expect("original payload");
        // The fixture came from the original bincode 1 writer, so re-encoding it is the
        // old-writer byte oracle.
        let mut reencoded = Vec::new();
        snapshot::encode_graph(
            &mut reencoded,
            &decoded.parts(),
            decoded.active.iter().copied(),
        )
        .expect("re-encode original payload");
        assert_eq!(
            reencoded, legacy,
            "codec must reproduce the old writer's bytes"
        );
        let graph = Hnsw::from_snapshot(decoded).expect("graph");
        assert!(graph.valid(2, Metric::L2, 16, 200));
        legacy[..8].copy_from_slice(&header.1.to_le_bytes());
        let mut bytes = Vec::new();
        snapshot::encode_header(&mut bytes, header).expect("header");
        bytes.extend_from_slice(&legacy);
        bytes.extend_from_slice(&crc32fast::hash(&bytes).to_le_bytes());
        std::fs::write(&snapshot, bytes).expect("install old snapshot");
        let mut store =
            crate::NativeStore::open(dir.path(), &OpenOptions::default()).expect("reopen legacy");
        assert_eq!(store.index_rebuilds(), 0);
        assert_eq!(
            store.knn("emb", &[3.0, 4.0], 1, 64).expect("search")[0].id,
            20
        );
        store
            .upsert_vectors("emb", &[(30, &[5.0, 6.0])])
            .expect("new write");
        store.flush().expect("new snapshot");
        let (_, current) = crate::load_snapshot(&snapshot).expect("new snapshot load");
        let mut bytes = Vec::new();
        current.encode_snapshot(&mut bytes).expect("new payload");
        let old = snapshot::decode_graph(bytes.as_slice(), bytes.len() as u64)
            .expect("old schema decoder");
        assert_eq!(
            (old.revision, old.dim, old.cosine, old.m, old.ef),
            (
                current.revision,
                current.dim,
                current.cosine,
                current.m,
                current.ef
            )
        );
        assert_eq!(old.data, current.data);
        assert_eq!(old.ids, current.ids);
        assert_eq!(old.links, current.links);
        assert_eq!(
            old.active.iter().copied().collect::<HashMap<_, _>>(),
            current.active
        );
        assert_eq!(old.active.len(), current.active.len());
        assert_eq!(old.entry, current.entry);
        assert_eq!(old.top, current.top);
        println!("BUILD_LEGACY old_snapshot_loaded=true rebuilds=0 new_payload_old_schema=true");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_retains_original_accumulator_rounding() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = quickdb_api::conformance::Lcg(97);
        for metric in [Metric::L2, Metric::Cosine] {
            for dim in [1, 7, 8, 9, 16, 31, 32, 127, 128, 129, 1024] {
                let mut h = Hnsw::new(dim, metric, 16, 200);
                h.avx2 = true;
                for _ in 0..100 {
                    let a = rng.vector(dim);
                    let b = rng.vector(dim);
                    assert_eq!(
                        h.distance(&a, &b).to_bits(),
                        h.distance_scalar(&a, &b).to_bits()
                    );
                }
                let a = vec![f32::MIN_POSITIVE; dim];
                let b = vec![-f32::MIN_POSITIVE; dim];
                assert_eq!(
                    h.distance(&a, &b).to_bits(),
                    h.distance_scalar(&a, &b).to_bits()
                );
            }
        }
        println!("BUILD_AVX2 scalar_bits_identical=true metrics=L2,Cosine dimensions=1..1024");
    }

    #[test]
    fn scratch_reuse_resets_and_nested_walks_are_isolated() {
        with_visited(1001, |outer| {
            assert!(outer.insert(1000));
            assert!(!outer.insert(1000));
            with_visited(1001, |inner| {
                assert!(inner.insert(1000));
                assert!(!inner.insert(1000));
            });
            assert!(!outer.insert(1000));
        });
        for nodes in [1, 65, 64, 1001] {
            with_visited(nodes, |visited| {
                for node in 0..nodes {
                    assert!(visited.insert(node));
                }
                for node in 0..nodes {
                    assert!(!visited.insert(node));
                }
            });
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn packed_l2_is_exact_and_falls_back_for_other_values() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = quickdb_api::conformance::Lcg(317);
        for dim in [1, 7, 8, 9, 16, 31, 32, 127, 128, 129, 256] {
            for _ in 0..40 {
                let a: Vec<_> = (0..dim).map(|_| (rng.next_u32() % 256) as f32).collect();
                let b: Vec<_> = (0..dim).map(|_| (rng.next_u32() % 256) as f32).collect();
                let mut h = Hnsw::new(dim, Metric::L2, 16, 200);
                h.append_vector(&a);
                h.append_vector(&b);
                assert!(h.packed.is_some());
                assert_eq!(
                    h.distance(h.vector(0), h.vector(1)).to_bits(),
                    h.distance_scalar(&a, &b).to_bits()
                );
                assert!(h.packed_slice(&a).is_none());
            }
            let mut h = Hnsw::new(dim, Metric::L2, 16, 200);
            h.append_vector(&vec![0.0; dim]);
            h.append_vector(&vec![255.0; dim]);
            assert_eq!(h.distance(h.vector(0), h.vector(1)), dim as f32 * 65025.0);
            let mut invalid = vec![0.0; dim];
            invalid[0] = 0.5;
            h.append_vector(&invalid);
            assert!(h.packed.is_none());
        }
        for value in [-1.0, 0.5, 256.0, f32::NAN, f32::INFINITY] {
            assert!(small_integer(value).is_none());
        }
        assert!(Hnsw::new(257, Metric::L2, 16, 200).packed.is_none());
        assert!(Hnsw::new(128, Metric::Cosine, 16, 200).packed.is_none());
        println!("BUILD_PACKED exact_bits=true dimensions=1..256 max_distance=16646400 invalid_values_fallback=true");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn packed_integer_build_matches_original_graph_for_one_two_seven_workers() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = quickdb_api::conformance::Lcg(419);
        let data: Vec<_> = (0..2048)
            .map(|id| {
                (
                    id,
                    (0..128)
                        .map(|_| (rng.next_u32() % 256) as f32)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let items: Vec<_> = data.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let mut original = Hnsw::new(128, Metric::L2, 16, 200);
        original.avx2 = false;
        original.packed = None;
        original.reuse_visited = false;
        original.prefetch = false;
        original.insert_many_with_workers(&items, 1);
        for workers in [1, 2, 7] {
            let mut h = Hnsw::new(128, Metric::L2, 16, 200);
            h.insert_many_with_workers(&items, workers);
            assert!(h.packed.is_some());
            assert_eq!(original.data, h.data);
            assert_eq!(original.links, h.links);
            assert_eq!(original.active, h.active);
            assert_eq!(original.entry, h.entry);
            assert_eq!(original.top, h.top);
            let mut bytes = Vec::new();
            h.encode_snapshot(&mut bytes).expect("snapshot");
            let graph = snapshot::decode_graph(bytes.as_slice(), bytes.len() as u64)
                .expect("reload packed cache");
            let restored = Hnsw::from_snapshot(graph).expect("restore graph");
            assert_eq!(h.packed, restored.packed);
            for (_, q) in data.iter().step_by(97) {
                assert_eq!(original.search(q, 10, 64), restored.search(q, 10, 64));
            }
        }
        println!("BUILD_PACKED_GRAPH workers=1,2,7 original_graph_identical=true snapshot_roundtrip=true");
    }
}

#[cfg(test)]
mod agy_repair_tests {
    use super::*;
    use crate::NativeStore;
    use quickdb_api::{
        conformance::{exact_distance, Lcg},
        Error, OpenOptions, Store, VectorIndexDef,
    };

    #[test]
    fn r4_search_filtered_clamps_huge_ef_and_k() {
        let mut h = Hnsw::new(2, Metric::L2, 8, 16);
        h.insert(1, &[0.0, 0.0]);
        h.insert(2, &[1.0, 0.0]);
        for (k, ef) in [
            (1, usize::MAX),
            (1, usize::MAX - 1),
            (usize::MAX, 1),
            (usize::MAX, usize::MAX),
        ] {
            let (hits, natural) =
                h.search_filtered(&[0.0, 0.0], k, ef, usize::MAX, 1, |_, out| out.fill(true));
            assert!(natural);
            assert_eq!(hits.len(), k.min(2));
            assert_eq!(hits[0].id, 1);
        }
    }

    #[test]
    fn r5_distance_scalar_matches_avx2_and_contract_on_unequal_lengths() {
        for metric in [Metric::L2, Metric::Cosine] {
            let h = Hnsw::new(16, metric, 8, 16);
            let short = [1.0f32; 9];
            let long = [0.0f32; 16];
            let expected = exact_distance(metric, &short, &long);
            assert_eq!(h.distance_scalar(&short, &long), expected);
            assert_eq!(h.distance_scalar(&long, &short), expected);
            assert_eq!(h.distance(&short, &long), h.distance_scalar(&short, &long));
            assert_eq!(h.distance(&long, &short), h.distance_scalar(&long, &short));
        }
    }

    #[test]
    fn r6_search_filtered_enforces_visit_budget_and_natural_flag() {
        let mut single = Hnsw::new(2, Metric::L2, 8, 16);
        single.insert(1, &[0.0, 0.0]);
        let (hits, natural) =
            single.search_filtered(&[0.0, 0.0], 1, 16, 0, 1, |_, out| out.fill(true));
        assert_eq!(hits.len(), 1);
        assert!(
            natural,
            "1-node index has no layer-0 neighbors, so budget=0 still finishes naturally"
        );

        let mut h = Hnsw::new(4, Metric::L2, 8, 32);
        let mut rng = Lcg(42);
        for id in 0..64 {
            h.insert(id, &rng.vector(4));
        }
        let q = rng.vector(4);
        let mut asked_with_zero = 0usize;
        let (hits0, natural0) = h.search_filtered(&q, 5, 16, 0, 4, |nodes, out| {
            // Only the entry node (before layer-0 expansion) may be checked when budget=0.
            asked_with_zero += nodes.len();
            out.fill(true);
        });
        assert!(
            !natural0,
            "budget=0 on multi-node graph must report natural=false"
        );
        assert_eq!(asked_with_zero, 1);
        assert_eq!(hits0.len(), 1);

        let mut asked_with_three = 0usize;
        let (_hits3, natural3) = h.search_filtered(&q, 5, 16, 3, 4, |nodes, out| {
            asked_with_three += nodes.len();
            out.fill(true);
        });
        assert!(!natural3, "budget=3 on 64-node graph must exhaust budget");
        // Entry (1) + at most 3 layer-0 neighbor distance evaluations.
        assert!(
            asked_with_three <= 4,
            "expected at most 1 + 3 nodes asked, got {asked_with_three}"
        );
    }

    #[test]
    fn r7_worker_panic_rolls_back_hnsw_batch_and_poisons_store_until_reopen() {
        let mut rng = Lcg(2026);
        let data: Vec<(u64, Vec<f32>)> = (0..320).map(|id| (id, rng.vector(4))).collect();
        let seed_items: Vec<(u64, &[f32])> = data[..256]
            .iter()
            .map(|(id, v)| (*id, v.as_slice()))
            .collect();
        let mut batch_data = data[256..320].to_vec();
        // Include a replacement of id 0 inside the panicking batch.
        batch_data[0].0 = 0;
        let batch_items: Vec<(u64, &[f32])> = batch_data
            .iter()
            .map(|(id, v)| (*id, v.as_slice()))
            .collect();

        for stage in [1u64, 2u64] {
            let mut h = Hnsw::new(4, Metric::L2, 8, 32);
            h.insert_many_with_workers(&seed_items, 2);
            let before_links = h.links.clone();
            let before_active = h.active.clone();

            PANIC_IN_PARALLEL_CALL.with(|c| c.set(stage));
            let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                h.insert_many_with_workers(&batch_items, 2);
            }));
            PANIC_IN_PARALLEL_CALL.with(|c| c.set(0));
            assert!(unwind.is_err(), "stage {stage} must panic");
            assert!(
                h.valid(4, Metric::L2, 8, 32),
                "Hnsw must remain valid after stage {stage} worker panic"
            );
            assert_eq!(h.len(), 256);
            assert_eq!(h.node_count(), 256);
            assert_eq!(h.active, before_active);
            assert_eq!(h.links, before_links);
        }

        // Store-level test: inject worker panic during store.upsert_vectors, verify reads and
        // flush are refused, and verify reopen rebuilds the committed vectors from SQLite.
        let root = std::env::var_os("QUICKDB_DATA")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let dir = tempfile::tempdir_in(root).expect("tempdir");
        let mut store = NativeStore::open(dir.path(), &OpenOptions::default()).expect("open");
        store
            .define_vector_index(&VectorIndexDef {
                name: "emb".into(),
                dim: 4,
                metric: Metric::L2,
                m: 8,
                ef_construction: 32,
            })
            .expect("define_vector_index");
        store
            .upsert_vectors("emb", &seed_items)
            .expect("seed upsert_vectors");
        store.flush().expect("initial flush");

        PANIC_IN_PARALLEL_CALL.with(|c| c.set(1));
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = store.upsert_vectors("emb", &batch_items);
        }));
        PANIC_IN_PARALLEL_CALL.with(|c| c.set(0));
        assert!(unwind.is_err());

        assert!(matches!(
            store.knn("emb", &data[0].1, 1, 32),
            Err(Error::Backend(_))
        ));
        assert!(matches!(store.count_vectors("emb"), Err(Error::Backend(_))));
        assert!(matches!(store.flush(), Err(Error::Backend(_))));
        drop(store);

        let reopened = NativeStore::open(dir.path(), &OpenOptions::default()).expect("reopen");
        assert_eq!(reopened.index_rebuilds(), 1);
        assert_eq!(reopened.count_vectors("emb").expect("count"), 319);
        let hits = reopened
            .knn("emb", &batch_data[0].1, 1, 64)
            .expect("knn after rebuild");
        assert_eq!(hits[0].id, 0);
        assert!(hits[0].distance < 1e-5);
    }
}

/// Measurement-only ABI: 0 reads, 1 resets, 2 increments the calling thread's count.
/// Without QUICKDB_DISTANCE_COUNTER at compile time the distance call is eliminated.
#[no_mangle]
pub extern "C" fn quickdb_distance_count(action: u8) -> u64 {
    thread_local! {
        static COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }
    COUNT.with(|count| {
        let value = count.get();
        match action {
            1 => count.set(0),
            2 => count.set(value.wrapping_add(1)),
            _ => {}
        }
        value
    })
}
