//! Standard ANN benchmark files, with independent truth for dataset prefixes.
use crate::{synthetic, Dataset};
use quickdb_api::{Error, Metric, Result};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

fn bad(message: impl Into<String>) -> Error {
    Error::Schema(message.into())
}

struct Records {
    input: BufReader<File>,
    dim: usize,
    count: usize,
}
impl Records {
    fn open(path: &Path) -> Result<Self> {
        let mut input = BufReader::new(File::open(path)?);
        let bytes = input.get_ref().metadata()?.len();
        let mut header = [0; 4];
        input
            .read_exact(&mut header)
            .map_err(|_| bad("missing vector dimension"))?;
        let dim = i32::from_le_bytes(header);
        if dim <= 0 {
            return Err(bad("vector dimension must be positive"));
        }
        let record_bytes = (dim as u64 + 1) * 4;
        if bytes % record_bytes != 0 {
            return Err(bad("truncated vector file or inconsistent record size"));
        }
        let count = usize::try_from(bytes / record_bytes).map_err(|_| bad("too many vectors"))?;
        // Reopen so the same record decoder validates the first dimension too.
        Ok(Self {
            input: BufReader::new(File::open(path)?),
            dim: dim as usize,
            count,
        })
    }

    fn record(&mut self, payload: &mut [u8]) -> Result<()> {
        let mut header = [0; 4];
        self.input
            .read_exact(&mut header)
            .map_err(|_| bad("truncated dimension"))?;
        if i32::from_le_bytes(header) != self.dim as i32 {
            return Err(bad("inconsistent dimensions"));
        }
        self.input
            .read_exact(payload)
            .map_err(|_| bad("truncated vector payload"))?;
        Ok(())
    }
}

fn floats(path: &Path, take: usize) -> Result<(Vec<Vec<f32>>, usize, usize)> {
    let mut records = Records::open(path)?;
    if take == 0 || take > records.count {
        return Err(bad("requested vector count must be positive and fit file"));
    }
    let mut payload = vec![0; records.dim * 4];
    let mut selected = Vec::with_capacity(take);
    for i in 0..records.count {
        records.record(&mut payload)?;
        let mut vector = if i < take {
            Vec::with_capacity(records.dim)
        } else {
            Vec::new()
        };
        for bytes in payload.chunks_exact(4) {
            let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            if !value.is_finite() {
                return Err(bad("non-finite vector value"));
            }
            if i < take {
                vector.push(value);
            }
        }
        if i < take {
            selected.push(vector);
        }
    }
    Ok((selected, records.dim, records.count))
}

fn truth(path: &Path, queries: usize, base_count: usize) -> Result<Vec<Vec<u64>>> {
    let mut records = Records::open(path)?;
    if records.dim < 10 || records.count < queries {
        return Err(bad(
            "truth needs at least ten neighbors and one record per query",
        ));
    }
    let mut payload = vec![0; records.dim * 4];
    let mut selected = Vec::with_capacity(queries);
    for i in 0..records.count {
        records.record(&mut payload)?;
        let mut ids = Vec::with_capacity(10);
        for bytes in payload.chunks_exact(4) {
            let id = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            if id < 0 || id as usize >= base_count {
                return Err(bad("truth id outside base file"));
            }
            if i < queries && ids.len() < 10 {
                if ids.contains(&(id as u64)) {
                    return Err(bad("duplicate ground truth neighbor"));
                }
                ids.push(id as u64);
            }
        }
        if i < queries {
            selected.push(ids);
        }
    }
    Ok(selected)
}

/// Loads zero-based ANN ids, retaining synthetic row and graph distributions.
/// Published ivecs truth is used only for the entire base file. Prefixes are
/// evaluated by the harness's independent, metric-aware exact scan instead.
pub fn ann(
    base: &Path,
    query: &Path,
    truth_path: &Path,
    limit: usize,
    queries: usize,
    metric: Metric,
) -> Result<Dataset> {
    let (vectors, dim, base_count) = floats(base, limit)?;
    let (query_vectors, query_dim, _) = floats(query, queries)?;
    if dim != query_dim {
        return Err(bad("query and base dimensions differ"));
    }
    // Validate supplied truth even when prefix truth will be computed independently.
    let published_truth = truth(truth_path, queries, base_count)?;
    let mut data = synthetic(limit, dim)?;
    for row in &mut data.rows {
        row.id -= 1;
    }
    for edge in &mut data.edges {
        edge.src -= 1;
        edge.dst -= 1;
    }
    data.vectors = vectors
        .into_iter()
        .enumerate()
        .map(|(id, vector)| (id as u64, vector))
        .collect();
    let metric_name = match metric {
        Metric::L2 => "l2",
        Metric::Cosine => "cosine",
        _ => return Err(bad("unsupported metric")),
    };
    data.metric = metric;
    data.query_vectors = Some(query_vectors);
    data.truth = (limit == base_count).then_some(published_truth);
    data.description = serde_json::json!({
        "type": "ann", "rows": limit, "dim": dim, "base_count": base_count,
        "queries": queries, "metric": metric_name,
        "base": base, "query": query, "truth": truth_path,
        "truth_source": if limit == base_count { "ivecs" } else { "exact_prefix" }
    });
    Ok(data)
}
