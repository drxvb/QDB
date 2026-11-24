//! Layer-0 reachability count and bridge repair.
//!
//! A node that no directed layer-0 path from the entry reaches can never be returned by the
//! graph search. Bridges are extra layer-0 out-links read only by search: they never count
//! against `m0`, are never pruned and are not written to the snapshot, so a reopened graph is
//! repaired again when it is loaded.
use super::Hnsw;
use std::collections::VecDeque;
use std::io::Write;

/// Bridges one node may hold.
pub(crate) const MAX_BRIDGES: usize = 4;
/// Selective policy: a local source is accepted only if its distance to the target is at most
/// this multiple of its own largest layer-0 link distance (in the index's distance units).
const LOCAL_RATIO: f32 = 1.35;
/// Breadth of the global layer-0 search that looks for a reachable source.
const REPAIR_EF: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Policy {
    Reference,
    Selective,
}

impl Policy {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "reference" => Some(Self::Reference),
            "selective" => Some(Self::Selective),
            _ => None,
        }
    }
    pub(crate) fn name(policy: Option<Self>) -> &'static str {
        match policy {
            None => "off",
            Some(Self::Reference) => "reference",
            Some(Self::Selective) => "selective",
        }
    }
}

/// `QUICKDB_REACH_REPAIR=off|reference|selective`; anything else, or unset, is off.
pub(crate) fn policy_from_env() -> Option<Policy> {
    std::env::var("QUICKDB_REACH_REPAIR")
        .ok()
        .and_then(|value| Policy::parse(&value))
}

/// Counts over live nodes only; tombstones are traversed but never counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Reach {
    pub live: usize,
    pub unreachable: usize,
    /// Live nodes that no layer-0 link points to (bridges excluded).
    pub zero_inbound: usize,
    /// Live nodes that neither a layer-0 link nor a bridge points to.
    pub zero_inbound_bridged: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Repair {
    pub before: Reach,
    pub after: Reach,
    pub bridges: usize,
    pub global_searches: usize,
    /// Selective only: targets bridged from their two-hop neighbourhood.
    pub local: usize,
    /// Targets whose global search found no candidate with spare capacity.
    pub fallback_any: usize,
    /// Targets left unbridged because no reachable node had capacity.
    pub failed: usize,
}

impl Hnsw {
    pub(crate) fn bridge_links(&self, node: usize) -> &[usize] {
        self.bridges.get(node).map_or(&[][..], Vec::as_slice)
    }

    fn out0(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.links[node][0]
            .iter()
            .chain(self.bridge_links(node))
            .copied()
    }

    fn spread(&self, start: usize, seen: &mut [bool]) {
        if seen[start] {
            return;
        }
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        while let Some(node) = queue.pop_front() {
            for next in self.out0(node) {
                if !seen[next] {
                    seen[next] = true;
                    queue.push_back(next);
                }
            }
        }
    }

    /// Positions reached from the entry along directed layer-0 links and bridges.
    pub(crate) fn reachable(&self) -> Vec<bool> {
        let mut seen = vec![false; self.ids.len()];
        if let Some(entry) = self.entry {
            self.spread(entry, &mut seen);
        }
        seen
    }

    pub(crate) fn reach(&self) -> Reach {
        let n = self.ids.len();
        let seen = self.reachable();
        let mut linked = vec![false; n];
        let mut bridged = vec![false; n];
        for (node, layers) in self.links.iter().enumerate() {
            for &next in &layers[0] {
                linked[next] = true;
            }
            for &next in self.bridge_links(node) {
                bridged[next] = true;
            }
        }
        let mut reach = Reach::default();
        for node in (0..n).filter(|&node| self.live(node)) {
            reach.live += 1;
            reach.unreachable += usize::from(!seen[node]);
            reach.zero_inbound += usize::from(!linked[node]);
            reach.zero_inbound_bridged += usize::from(!linked[node] && !bridged[node]);
        }
        reach
    }

    fn has_capacity(&self, node: usize) -> bool {
        self.bridge_links(node).len() < MAX_BRIDGES
    }

    fn global_source(&self, target: usize, entry: usize, seen: &[bool]) -> Option<usize> {
        // The walk starts at the entry on layer 0, so every node it returns is reachable.
        self.layer(self.vector(target), entry, REPAIR_EF, 0)
            .into_iter()
            .map(|c| c.node)
            .find(|&node| node != target && seen[node] && self.has_capacity(node))
    }

    fn any_source(&self, target: usize, seen: &[bool]) -> Option<usize> {
        (0..self.ids.len()).find(|&node| node != target && seen[node] && self.has_capacity(node))
    }

    fn local_source(&self, target: usize, seen: &[bool]) -> Option<usize> {
        let mut best: Option<(f32, usize)> = None;
        let mut consider = |node: usize| {
            if node == target || !seen[node] || !self.has_capacity(node) {
                return;
            }
            let distance = self.distance(self.vector(node), self.vector(target));
            if best.is_none_or(|(d, n)| distance.total_cmp(&d).then(node.cmp(&n)).is_lt()) {
                best = Some((distance, node));
            }
        };
        for first in self.out0(target) {
            consider(first);
            for second in self.out0(first) {
                consider(second);
            }
        }
        let (distance, source) = best?;
        let radius = self.links[source][0]
            .iter()
            .map(|&n| self.distance(self.vector(source), self.vector(n)))
            .reduce(f32::max)?;
        (distance <= LOCAL_RATIO * radius).then_some(source)
    }

    /// Adds bridges until every live node is reachable from the entry on layer 0. Targets are
    /// taken in node order, and each bridge marks everything it makes reachable before the
    /// next target, so one bridge often repairs a whole unreachable cluster.
    pub(crate) fn repair_reach(&mut self, policy: Policy) -> Repair {
        let before = self.reach();
        let mut out = Repair {
            before,
            after: before,
            ..Repair::default()
        };
        let Some(entry) = self.entry else {
            return out;
        };
        if before.unreachable == 0 {
            return out;
        }
        let n = self.ids.len();
        if self.bridges.len() < n {
            self.bridges.resize(n, Vec::new());
        }
        let mut seen = self.reachable();
        for target in 0..n {
            if seen[target] || !self.live(target) {
                continue;
            }
            let mut source = None;
            if policy == Policy::Selective {
                source = self.local_source(target, &seen);
                out.local += usize::from(source.is_some());
            }
            if source.is_none() {
                out.global_searches += 1;
                source = self.global_source(target, entry, &seen);
            }
            if source.is_none() {
                source = self.any_source(target, &seen);
                out.fallback_any += usize::from(source.is_some());
            }
            let Some(source) = source else {
                out.failed += 1;
                continue;
            };
            self.bridges[source].push(target);
            out.bridges += 1;
            self.spread(target, &mut seen);
        }
        out.after = self.reach();
        out
    }

    /// Measurement hook, cut every inbound link and bridge of `id` on all layers (never the
    /// entry).
    fn cut_inbound(&mut self, id: u64) {
        if let Some(&node) = self.active.get(&id) {
            self.strand(&[node]);
        }
    }

    /// Measurement hook, cuts every inbound link and bridge of these positions on all layers;
    /// the entry is skipped. Returns the positions that were cut, ascending.
    pub(crate) fn strand(&mut self, nodes: &[usize]) -> Vec<usize> {
        let n = self.ids.len();
        let mut cut = vec![false; n];
        for &node in nodes {
            if node < n && self.entry != Some(node) {
                cut[node] = true;
            }
        }
        for layers in &mut self.links {
            for links in layers.iter_mut() {
                links.retain(|&next| !cut[next]);
            }
        }
        for bridges in &mut self.bridges {
            bridges.retain(|&next| !cut[next]);
        }
        (0..n).filter(|&node| cut[node]).collect()
    }

    /// Runs when an index finishes building: at the end of every `insert_many` (bulk load,
    /// `upsert_vectors` and the compaction rebuild in `flush`) and after a snapshot load. The
    /// environment is read once per call. With repair off and no measurement hook set, nothing
    /// changes.
    /// Measurement hooks: `QUICKDB_REACH_PLANT=<id>` cuts that node's inbound links first, and
    /// `QUICKDB_REACH_REPORT=<path>` appends one JSON line of counts.
    pub(crate) fn reach_after_build(&mut self) {
        let policy = policy_from_env();
        let plant = std::env::var("QUICKDB_REACH_PLANT")
            .ok()
            .and_then(|v| v.parse::<u64>().ok());
        let report = std::env::var_os("QUICKDB_REACH_REPORT");
        if policy.is_none() && plant.is_none() && report.is_none() {
            return;
        }
        if let Some(id) = plant {
            self.cut_inbound(id);
        }
        let result = match policy {
            Some(policy) => self.repair_reach(policy),
            None if report.is_some() => {
                let reach = self.reach();
                Repair {
                    before: reach,
                    after: reach,
                    ..Repair::default()
                }
            }
            None => return,
        };
        if let Some(path) = report {
            let line = format!(
                "{{\"policy\":\"{}\",\"live\":{},\"unreachable_before\":{},\"unreachable_after\":{},\"zero_inbound\":{},\"zero_inbound_bridged_after\":{},\"bridges\":{},\"bridges_total\":{},\"global_searches\":{},\"local\":{},\"fallback_any\":{},\"failed\":{}}}",
                Policy::name(policy),
                result.after.live,
                result.before.unreachable,
                result.after.unreachable,
                result.after.zero_inbound,
                result.after.zero_inbound_bridged,
                result.bridges,
                self.bridges.iter().map(Vec::len).sum::<usize>(),
                result.global_searches,
                result.local,
                result.fallback_any,
                result.failed,
            );
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = writeln!(file, "{line}");
            }
        }
    }
}
