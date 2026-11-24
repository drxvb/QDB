//! Power-loss emulation, shared by `tests/powerloss.rs` and `examples/pl_rebuild.rs`.
//! Parses the log of `tools/fsrecord.c` and builds the directory a power cut at a chosen record
//! could leave behind. Model (stricter than ext4, as POSIX allows):
//! - file content survives as of the file's last `fsync`/`fdatasync`; later writes and truncates
//!   are lost (`Drop`), or an arbitrary subset of them lands (`Reorder`), or an arbitrary subset of
//!   their 4 KiB-aligned pieces lands (`Torn`);
//! - names (create, rename, unlink) survive as of the directory's last `fsync`; after it, a prefix
//!   of the later name operations lands (none for `Drop`); the directory itself always exists;
//! - `-shm` files are left out: SQLite rebuilds the WAL index from the WAL and never needs it.
// Each user takes a different subset.
#![allow(dead_code)]

use quickdb_api::conformance::Lcg;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::Path;

pub const OPEN: u8 = 1;
pub const WRITE: u8 = 2;
pub const TRUNC: u8 = 3;
pub const RENAME: u8 = 4;
pub const UNLINK: u8 = 5;
pub const FSYNC: u8 = 6;
pub const FDATASYNC: u8 = 7;
pub const CLOSE: u8 = 8;
pub const MARK: u8 = 9;
pub const PWRITE: u8 = 10;
const O_TRUNC: u64 = 0o1000;
const PIECE: u64 = 4096;

pub fn op_name(op: u8) -> &'static str {
    match op {
        OPEN => "open",
        WRITE => "write",
        TRUNC => "ftruncate",
        RENAME => "rename",
        UNLINK => "unlink",
        FSYNC => "fsync",
        FDATASYNC => "fdatasync",
        CLOSE => "close",
        MARK => "stdout",
        PWRITE => "pwrite",
        _ => "?",
    }
}

#[derive(Debug, Clone)]
pub struct Rec {
    pub op: u8,
    pub fd: i32,
    pub name: String,
    pub name2: String,
    /// Open flags, write offset or truncate length.
    pub a: u64,
    pub data: Vec<u8>,
}

impl Rec {
    pub fn describe(&self) -> String {
        let name = if self.name.is_empty() && self.op != MARK {
            "<dir>"
        } else {
            &self.name
        };
        match self.op {
            WRITE | PWRITE => format!(
                "{} {name} off={} len={}",
                op_name(self.op),
                self.a,
                self.data.len()
            ),
            TRUNC => format!("ftruncate {name} len={}", self.a),
            RENAME => format!("rename {name} -> {}", self.name2),
            MARK => format!(
                "stdout {:?}",
                String::from_utf8_lossy(&self.data).trim_end()
            ),
            OPEN => format!("open {name} flags={:#o}", self.a),
            _ => format!("{} {name}", op_name(self.op)),
        }
    }
}

fn take<'a>(b: &'a [u8], at: &mut usize, n: usize) -> Result<&'a [u8], String> {
    let end = at
        .checked_add(n)
        .filter(|e| *e <= b.len())
        .ok_or("log truncated")?;
    let s = &b[*at..end];
    *at = end;
    Ok(s)
}

pub fn parse(b: &[u8]) -> Result<Vec<Rec>, String> {
    let mut at = 0;
    let mut out = Vec::new();
    let u16_at = |at: &mut usize| -> Result<usize, String> {
        Ok(u16::from_le_bytes(take(b, at, 2)?.try_into().map_err(|_| "u16")?) as usize)
    };
    let u64_at = |at: &mut usize| -> Result<u64, String> {
        Ok(u64::from_le_bytes(
            take(b, at, 8)?.try_into().map_err(|_| "u64")?,
        ))
    };
    while at < b.len() {
        let op = take(b, &mut at, 1)?[0];
        let fd = i32::from_le_bytes(take(b, &mut at, 4)?.try_into().map_err(|_| "fd")?);
        let la = u16_at(&mut at)?;
        let name = String::from_utf8_lossy(take(b, &mut at, la)?).into_owned();
        let lb = u16_at(&mut at)?;
        let name2 = String::from_utf8_lossy(take(b, &mut at, lb)?).into_owned();
        let a = u64_at(&mut at)?;
        let len = u64_at(&mut at)?;
        let data = if matches!(op, WRITE | PWRITE | MARK) {
            take(b, &mut at, usize::try_from(len).map_err(|_| "length")?)?.to_vec()
        } else {
            Vec::new()
        };
        if !(OPEN..=PWRITE).contains(&op) {
            return Err(format!("unknown op {op} at record {}", out.len()));
        }
        out.push(Rec {
            op,
            fd,
            name,
            name2,
            a,
            data,
        });
    }
    Ok(out)
}

/// The complete stdout lines of the writer, each with the index of the record that completed it.
pub fn lines(recs: &[Rec]) -> Vec<(usize, String)> {
    let mut buf = Vec::new();
    let mut out = Vec::new();
    for (i, r) in recs.iter().enumerate().filter(|(_, r)| r.op == MARK) {
        buf.extend_from_slice(&r.data);
        while let Some(p) = buf.iter().position(|c| *c == b'\n') {
            let line: Vec<u8> = buf.drain(..=p).collect();
            out.push((i, String::from_utf8_lossy(&line).trim_end().to_string()));
        }
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Drop,
    Reorder,
    Torn,
}

impl Mode {
    pub const ALL: [Mode; 3] = [Mode::Drop, Mode::Reorder, Mode::Torn];
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "drop" => Some(Mode::Drop),
            "reorder" => Some(Mode::Reorder),
            "torn" => Some(Mode::Torn),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Mode::Drop => "drop",
            Mode::Reorder => "reorder",
            Mode::Torn => "torn",
        }
    }
}

#[derive(Debug, Clone)]
enum Pending {
    Write(u64, Vec<u8>, usize),
    Trunc(u64, usize),
}

#[derive(Debug, Clone)]
enum NsOp {
    Link(String, usize, usize),
    Rename(String, String, usize),
    Unlink(String, usize),
}

#[derive(Debug, Default, Clone)]
struct Inode {
    durable: Vec<u8>,
    pending: Vec<Pending>,
}

fn apply(buf: &mut Vec<u8>, p: &Pending) {
    match p {
        Pending::Write(off, d, _) => write_at(buf, *off, d),
        Pending::Trunc(len, _) => buf.resize(*len as usize, 0),
    }
}

fn write_at(buf: &mut Vec<u8>, off: u64, d: &[u8]) {
    let off = off as usize;
    if buf.len() < off + d.len() {
        buf.resize(off + d.len(), 0);
    }
    buf[off..off + d.len()].copy_from_slice(d);
}

fn ns_apply(ns: &mut BTreeMap<String, usize>, op: &NsOp) {
    match op {
        NsOp::Link(n, i, _) => {
            ns.insert(n.clone(), *i);
        }
        NsOp::Rename(a, b, _) => {
            if let Some(i) = ns.remove(a) {
                ns.insert(b.clone(), i);
            }
        }
        NsOp::Unlink(n, _) => {
            ns.remove(n);
        }
    }
}

/// File-system state after replaying a prefix of the log.
#[derive(Debug, Default, Clone)]
pub struct Fs {
    inodes: Vec<Inode>,
    /// `None` is the watched directory.
    fds: HashMap<i32, Option<usize>>,
    ns: BTreeMap<String, usize>,
    durable_ns: BTreeMap<String, usize>,
    ns_pending: Vec<NsOp>,
    /// Records replayed so far.
    pub at: usize,
}

impl Fs {
    pub fn step(&mut self, r: &Rec) {
        let at = self.at;
        self.at += 1;
        let ino = |fds: &HashMap<i32, Option<usize>>| fds.get(&r.fd).copied().flatten();
        match r.op {
            OPEN if r.name.is_empty() => {
                self.fds.insert(r.fd, None);
            }
            OPEN => {
                let i = match self.ns.get(&r.name) {
                    Some(&i) => i,
                    None => {
                        self.inodes.push(Inode::default());
                        let i = self.inodes.len() - 1;
                        self.ns.insert(r.name.clone(), i);
                        self.ns_pending.push(NsOp::Link(r.name.clone(), i, at));
                        i
                    }
                };
                if r.a & O_TRUNC != 0 {
                    self.inodes[i].pending.push(Pending::Trunc(0, at));
                }
                self.fds.insert(r.fd, Some(i));
            }
            WRITE | PWRITE => {
                if let Some(i) = ino(&self.fds) {
                    self.inodes[i]
                        .pending
                        .push(Pending::Write(r.a, r.data.clone(), at));
                }
            }
            TRUNC => {
                if let Some(i) = ino(&self.fds) {
                    self.inodes[i].pending.push(Pending::Trunc(r.a, at));
                }
            }
            FSYNC | FDATASYNC => match self.fds.get(&r.fd) {
                Some(None) => {
                    self.durable_ns = self.ns.clone();
                    self.ns_pending.clear();
                }
                Some(Some(i)) => {
                    let node = &mut self.inodes[*i];
                    for p in std::mem::take(&mut node.pending) {
                        apply(&mut node.durable, &p);
                    }
                }
                None => {}
            },
            CLOSE => {
                self.fds.remove(&r.fd);
            }
            RENAME => {
                if let Some(i) = self.ns.remove(&r.name) {
                    self.ns.insert(r.name2.clone(), i);
                }
                self.ns_pending
                    .push(NsOp::Rename(r.name.clone(), r.name2.clone(), at));
            }
            UNLINK => {
                self.ns.remove(&r.name);
                self.ns_pending.push(NsOp::Unlink(r.name.clone(), at));
            }
            _ => {}
        }
    }

    /// The files a power cut now could leave, and a description of every unsynced operation
    /// with whether it landed (record index, `+` landed, `-` lost, `~` torn).
    pub fn image(&self, mode: Mode, rng: &mut Lcg) -> (BTreeMap<String, Vec<u8>>, String) {
        let mut how = String::new();
        let coin = |rng: &mut Lcg| mode != Mode::Drop && rng.next_u32() & 1 == 1;
        let mut ns = self.durable_ns.clone();
        let k = match mode {
            Mode::Drop => 0,
            _ => rng.next_u32() as usize % (self.ns_pending.len() + 1),
        };
        for (j, op) in self.ns_pending.iter().enumerate() {
            let at = match op {
                NsOp::Link(_, _, a) | NsOp::Rename(_, _, a) | NsOp::Unlink(_, a) => a,
            };
            let _ = write!(how, " {}{at}", if j < k { '+' } else { '-' });
            if j < k {
                ns_apply(&mut ns, op);
            }
        }
        let mut contents: HashMap<usize, Vec<u8>> = HashMap::new();
        let mut files = BTreeMap::new();
        for (name, &i) in &ns {
            if name.ends_with("-shm") {
                continue;
            }
            if let std::collections::hash_map::Entry::Vacant(slot) = contents.entry(i) {
                let node = &self.inodes[i];
                let mut buf = node.durable.clone();
                for p in &node.pending {
                    match (mode, p) {
                        (Mode::Torn, Pending::Write(off, d, at)) => {
                            let (mut lo, end) = (*off, *off + d.len() as u64);
                            let (mut kept, mut lost) = (0, 0);
                            while lo < end {
                                let hi = ((lo / PIECE + 1) * PIECE).min(end);
                                if coin(rng) {
                                    let s = (lo - off) as usize..(hi - off) as usize;
                                    write_at(&mut buf, lo, &d[s]);
                                    kept += 1;
                                } else {
                                    lost += 1;
                                }
                                lo = hi;
                            }
                            let c = match (kept, lost) {
                                (_, 0) => '+',
                                (0, _) => '-',
                                _ => '~',
                            };
                            let _ = write!(how, " {c}{at}");
                        }
                        (_, p) => {
                            let at = match p {
                                Pending::Write(_, _, a) | Pending::Trunc(_, a) => a,
                            };
                            let keep = coin(rng);
                            let _ = write!(how, " {}{at}", if keep { '+' } else { '-' });
                            if keep {
                                apply(&mut buf, p);
                            }
                        }
                    }
                }
                slot.insert(buf);
            }
            files.insert(name.clone(), contents[&i].clone());
        }
        (files, how)
    }
}

pub fn write_image(dir: &Path, files: &BTreeMap<String, Vec<u8>>) -> std::io::Result<()> {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir)?;
    for (name, data) in files {
        std::fs::write(dir.join(name), data)?;
    }
    Ok(())
}

/// What the writer had reported before record `c`: per kind the last committed batch, the last
/// flushed batch.
pub fn progress(lines: &[(usize, String)], c: usize) -> ([Option<u64>; 3], Option<u64>) {
    let mut committed = [None; 3];
    let mut flushed = None;
    for (_, l) in lines.iter().take_while(|(i, _)| *i < c) {
        let w: Vec<&str> = l.split_whitespace().collect();
        let n = w.get(1).and_then(|x| x.parse::<u64>().ok());
        match (w.first().copied(), n) {
            (Some("COMMITTED"), Some(n)) => {
                let k = match w.get(2).copied() {
                    Some("rows") => 0,
                    Some("vectors") => 1,
                    _ => 2,
                };
                committed[k] = Some(n);
            }
            (Some("FLUSHED"), Some(n)) => flushed = Some(n),
            _ => {}
        }
    }
    (committed, flushed)
}
