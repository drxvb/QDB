//! Version 3 wire format: fixed-width little-endian words, byte tags and u64 lengths.
//! Kept independent of the store so format probes and mutation tests use the same parser.
use std::io::{self, Read, Write};

pub const MAGIC: u64 = 0x514442484e535733;
pub type Header = ([u8; 16], u64, i64);

fn invalid() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid snapshot payload")
}

pub struct GraphParts<'a> {
    pub revision: u64,
    pub dim: usize,
    pub cosine: bool,
    pub m: usize,
    pub ef: usize,
    pub data: &'a [f32],
    pub ids: &'a [u64],
    pub links: &'a [Vec<Vec<usize>>],
    pub entry: Option<usize>,
    pub top: usize,
}

#[derive(Debug, PartialEq)]
pub struct Graph {
    pub revision: u64,
    pub dim: usize,
    pub cosine: bool,
    pub m: usize,
    pub ef: usize,
    pub data: Vec<f32>,
    pub ids: Vec<u64>,
    pub links: Vec<Vec<Vec<usize>>>,
    // A sequence preserves the writer's actual map iteration order, not a sorted order.
    pub active: Vec<(u64, usize)>,
    pub entry: Option<usize>,
    pub top: usize,
}
impl Graph {
    pub fn parts(&self) -> GraphParts<'_> {
        GraphParts {
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
        }
    }

    pub fn valid(&self) -> bool {
        if self.dim == 0
            || self.m < 2
            || self.ef < self.m.max(16)
            || self.ids.len().checked_mul(self.dim) != Some(self.data.len())
            || self.links.len() != self.ids.len()
            || self.top > 24
            || self.entry.is_none() != self.ids.is_empty()
            || self.data.iter().any(|v| !v.is_finite())
            || self.ids.iter().any(|&id| id > i64::MAX as u64)
            || self.active.len() > self.ids.len()
        {
            return false;
        }
        if let Some(n) = self.entry {
            if self.links.get(n).map(Vec::len) != Some(self.top + 1) {
                return false;
            }
        }
        for (node, layers) in self.links.iter().enumerate() {
            let mut hash = (node as u64).wrapping_add(0x9e3779b97f4a7c15);
            hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d049bb133111eb);
            hash ^= hash >> 31;
            let mut level = 0;
            while hash.is_multiple_of(self.m as u64) && level < 24 {
                level += 1;
                hash /= self.m as u64;
            }
            if layers.len() != level + 1 || layers.len() > self.top + 1 {
                return false;
            }
            for (level, neighbors) in layers.iter().enumerate() {
                if neighbors
                    .iter()
                    .any(|&n| self.links.get(n).is_none_or(|ls| ls.len() <= level))
                {
                    return false;
                }
            }
        }
        self.active
            .iter()
            .all(|&(id, n)| self.ids.get(n) == Some(&id))
    }
}

pub fn word(out: &mut impl Write, value: u64) -> io::Result<()> {
    out.write_all(&value.to_le_bytes())
}
fn option(out: &mut impl Write, value: Option<usize>) -> io::Result<()> {
    out.write_all(&[u8::from(value.is_some())])?;
    if let Some(value) = value {
        word(out, value as u64)?;
    }
    Ok(())
}
pub fn encode_header(out: &mut impl Write, header: Header) -> io::Result<()> {
    word(out, MAGIC)?;
    out.write_all(&header.0)?;
    word(out, header.1)?;
    out.write_all(&header.2.to_le_bytes())
}
pub fn encode_graph(
    out: &mut impl Write,
    graph: &GraphParts<'_>,
    active: impl ExactSizeIterator<Item = (u64, usize)>,
) -> io::Result<()> {
    word(out, graph.revision)?;
    word(out, graph.dim as u64)?;
    out.write_all(&[u8::from(graph.cosine)])?;
    word(out, graph.m as u64)?;
    word(out, graph.ef as u64)?;
    word(out, graph.data.len() as u64)?;
    for value in graph.data {
        out.write_all(&value.to_bits().to_le_bytes())?;
    }
    word(out, graph.ids.len() as u64)?;
    for &id in graph.ids {
        word(out, id)?;
    }
    encode_links(out, graph.links)?;
    word(out, active.len() as u64)?;
    for (id, n) in active {
        word(out, id)?;
        word(out, n as u64)?;
    }
    option(out, graph.entry)?;
    word(out, graph.top as u64)
}
fn encode_links(out: &mut impl Write, links: &[Vec<Vec<usize>>]) -> io::Result<()> {
    word(out, links.len() as u64)?;
    for layers in links {
        word(out, layers.len() as u64)?;
        for neighbors in layers {
            word(out, neighbors.len() as u64)?;
            for &n in neighbors {
                word(out, n as u64)?;
            }
        }
    }
    Ok(())
}
// The probe's historical graph-only digest excludes revision, floats and randomized map order.
pub fn encode_digest(out: &mut impl Write, graph: &GraphParts<'_>) -> io::Result<()> {
    word(out, graph.dim as u64)?;
    out.write_all(&[u8::from(graph.cosine)])?;
    word(out, graph.m as u64)?;
    word(out, graph.ef as u64)?;
    word(out, graph.ids.len() as u64)?;
    for &id in graph.ids {
        word(out, id)?;
    }
    encode_links(out, graph.links)?;
    option(out, graph.entry)?;
    word(out, graph.top as u64)
}

struct Decoder<R> {
    input: R,
    remaining: u64,
    allocation: u64,
}
impl<R: Read> Decoder<R> {
    fn bytes<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        self.remaining = self.remaining.checked_sub(N as u64).ok_or_else(invalid)?;
        let mut bytes = [0; N];
        self.input.read_exact(&mut bytes)?;
        Ok(bytes)
    }
    fn word(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn size(&mut self) -> io::Result<usize> {
        usize::try_from(self.word()?).map_err(|_| invalid())
    }
    fn tag(&mut self) -> io::Result<bool> {
        match self.bytes::<1>()?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid()),
        }
    }
    fn vector<T>(
        &mut self,
        minimum_wire_size: u64,
        mut element: impl FnMut(&mut Self) -> io::Result<T>,
    ) -> io::Result<Vec<T>> {
        let length = self.size()?;
        let wire_bytes = (length as u64)
            .checked_mul(minimum_wire_size)
            .ok_or_else(invalid)?;
        if wire_bytes > self.remaining {
            return Err(invalid());
        }
        let allocation = length
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(invalid)?;
        self.allocation = self
            .allocation
            .checked_sub(allocation as u64)
            .ok_or_else(invalid)?;
        let mut values = Vec::new();
        values.try_reserve_exact(length).map_err(|_| invalid())?;
        for _ in 0..length {
            values.push(element(self)?);
        }
        Ok(values)
    }
    fn graph(&mut self) -> io::Result<Graph> {
        let revision = self.word()?;
        let dim = self.size()?;
        let cosine = self.tag()?;
        let m = self.size()?;
        let ef = self.size()?;
        if dim == 0 || m < 2 || ef < m.max(16) {
            return Err(invalid());
        }
        let data = self.vector(4, |d| Ok(f32::from_bits(u32::from_le_bytes(d.bytes()?))))?;
        let ids = self.vector(8, Self::word)?;
        let links = self.vector(8, |d| d.vector(8, |d| d.vector(8, Self::size)))?;
        let active = self.vector(16, |d| Ok((d.word()?, d.size()?)))?;
        let entry = if self.tag()? {
            Some(self.size()?)
        } else {
            None
        };
        let top = self.size()?;
        let graph = Graph {
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
        };
        if !graph.valid() {
            return Err(invalid());
        }
        let key_bytes = (graph.active.len() as u64)
            .checked_mul(32)
            .ok_or_else(invalid)?;
        self.allocation = self.allocation.checked_sub(key_bytes).ok_or_else(invalid)?;
        let mut keys = std::collections::HashSet::new();
        keys.try_reserve(graph.active.len())
            .map_err(|_| invalid())?;
        if graph.active.iter().any(|&(id, _)| !keys.insert(id)) {
            return Err(invalid());
        }
        Ok(graph)
    }
}
fn decoder<R: Read>(input: R, length: u64) -> Decoder<R> {
    // Nested Vec headers use 24 native bytes for an 8-byte wire length. Eight times
    // the payload bounds all parser allocations, including empty nested containers.
    Decoder {
        input,
        remaining: length,
        allocation: length.saturating_mul(8),
    }
}
pub fn decode_graph(input: impl Read, length: u64) -> io::Result<Graph> {
    let mut decoder = decoder(input, length);
    let graph = decoder.graph()?;
    if decoder.remaining != 0 {
        return Err(invalid());
    }
    Ok(graph)
}
pub fn decode(input: impl Read, length: u64) -> io::Result<(Header, Graph)> {
    let mut decoder = decoder(input, length);
    if decoder.word()? != MAGIC {
        return Err(invalid());
    }
    let id = decoder.bytes()?;
    let revision = decoder.word()?;
    let nonce = i64::from_le_bytes(decoder.bytes()?);
    let graph = decoder.graph()?;
    if decoder.remaining != 0 {
        return Err(invalid());
    }
    Ok(((id, revision, nonce), graph))
}
