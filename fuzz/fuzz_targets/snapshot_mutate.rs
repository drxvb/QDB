//! Valid small snapshots, damaged by flipping, overwriting, inserting, deleting and truncating
//! bytes, then checked and decoded the way the store reads them (CRC32, then decode).
#![no_main]
use libfuzzer_sys::fuzz_target;

#[allow(dead_code, unused_imports)]
#[path = "../../crates/quickdb-native/src/hnsw.rs"]
mod hnsw;

include!("snapshot_common.rs");

const FIXTURES: [&[u8]; 4] = [
    include_bytes!("../../crates/quickdb-native/tests/fixtures/snap/small.bin"),
    include_bytes!("../../crates/quickdb-native/tests/fixtures/snap/deletions.bin"),
    include_bytes!("../../crates/quickdb-native/tests/fixtures/snap/empty.bin"),
    include_bytes!("../../crates/quickdb-native/tests/fixtures/snap/cosine.bin"),
];

fn mutate(mut bytes: Vec<u8>, ops: &[u8]) -> Vec<u8> {
    for op in ops.chunks_exact(4) {
        let at = u16::from_le_bytes([op[1], op[2]]) as usize;
        let value = op[3];
        let len = bytes.len();
        match op[0] % 6 {
            0 if len > 0 => bytes[at % len] ^= 1 << (value % 8),
            1 if len > 0 => bytes[at % len] = value,
            2 => bytes.truncate(at % (len + 1)),
            3 => bytes.insert(at % (len + 1), value),
            4 if len > 0 => {
                let start = at % len;
                let end = (start + value as usize).min(len);
                bytes.drain(start..end);
            }
            5 if len >= 8 => {
                // Overwrite one aligned little-endian word, where lengths and counts live.
                let start = (at % (len / 8)) * 8;
                let word = u64::from(value).wrapping_mul(0x0101_0101_0101_0101) >> (at % 64);
                bytes[start..start + 8].copy_from_slice(&word.to_le_bytes());
            }
            _ => {}
        }
    }
    bytes
}

fuzz_target!(|data: &[u8]| {
    let Some((&control, ops)) = data.split_first() else {
        return;
    };
    let fixture = FIXTURES[control as usize % FIXTURES.len()];
    let mut bytes = mutate(fixture.to_vec(), ops);
    let Some(payload_len) = bytes.len().checked_sub(4) else {
        return;
    };
    if control & 0x80 != 0 {
        // Re-seal the damaged payload so the decoder, not the checksum, has to reject it.
        let crc = crc32fast::hash(&bytes[..payload_len]);
        bytes[payload_len..].copy_from_slice(&crc.to_le_bytes());
    }
    let stored = u32::from_le_bytes(bytes[payload_len..].try_into().expect("4 bytes"));
    if crc32fast::hash(&bytes[..payload_len]) != stored {
        return;
    }
    let payload = &bytes[..payload_len];
    if let Ok((_, graph)) = snapshot::decode(payload, payload.len() as u64) {
        exercise(graph, control);
    }
});
