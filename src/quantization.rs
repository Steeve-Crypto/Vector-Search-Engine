//! Scalar + real Product Quantization (PQ) with k-means (Phase 6/8).
//!
//! - Scalar quant: fast 4x memory reduction (u8 per dimension).
//! - Real PQ: trained codebooks using k-means on subvectors. Dramatically
//!   better compression (M bytes for M subvectors) with controllable error.
//!
//! Both paths support dequantize for use with HNSW (f32 required).
//! quant_error() and PQ::quantization_error() help with monitoring.

/// Quantize a normalized f32 vector to u8 (assuming values in ~[-1, 1]).
pub fn quantize(vec: &[f32]) -> Vec<u8> {
    vec.iter()
        .map(|&x| {
            let scaled = (x + 1.0) * 127.5;
            scaled.clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Dequantize u8 back to f32 approx [-1, 1].
pub fn dequantize(q: &[u8]) -> Vec<f32> {
    q.iter()
        .map(|&b| (b as f32 / 127.5) - 1.0)
        .collect()
}

/// Compute RMS quantization error (root mean squared diff) between original and dequantized.
/// Useful for observability / histograms. Typical values ~0.003-0.01 for normalized vectors.
pub fn quantization_error(orig: &[f32]) -> f64 {
    if orig.is_empty() {
        return 0.0;
    }
    let q = quantize(orig);
    let dq = dequantize(&q);
    let mut sum_sq = 0.0f32;
    for (o, d) in orig.iter().zip(dq.iter()) {
        let diff = o - d;
        sum_sq += diff * diff;
    }
    (sum_sq / orig.len() as f32).sqrt() as f64
}

/// Simple quantized storage example (e.g., for sled value).
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct QuantizedVector {
    pub data: Vec<u8>,
    pub original_dim: usize,
}

impl QuantizedVector {
    pub fn from_vec(vec: &[f32]) -> Self {
        Self {
            data: quantize(vec),
            original_dim: vec.len(),
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        dequantize(&self.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_dequant() {
        let original = vec![-1.0, 0.0, 0.5, 1.0];
        let q = quantize(&original);
        let d = dequantize(&q);
        assert_eq!(d.len(), 4);
        // Approx equal
        for (o, dd) in original.iter().zip(d.iter()) {
            assert!((o - dd).abs() < 0.01);
        }
    }

    #[test]
    fn test_quantized_vector() {
        let v = vec![0.1, -0.2, 0.9];
        let qv = QuantizedVector::from_vec(&v);
        let back = qv.to_vec();
        assert_eq!(back.len(), 3);
    }

    #[test]
    fn test_pq_train_quant_dequant() {
        // Use synthetic normalized-ish vectors for training
        let mut samples = vec![];
        for i in 0..200 {
            let mut v = vec![0.0f32; 16]; // small dim for test speed
            for j in 0..16 {
                v[j] = ((i * 7 + j * 3) % 200) as f32 / 100.0 - 1.0;
            }
            // quick normalize
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for x in &mut v { *x /= norm; }
            }
            samples.push(v);
        }

        let pq = ProductQuantizer::train(&samples, 4, 16); // m=4, small k=16 for test
        assert!(pq.is_trained());
        assert_eq!(pq.m, 4);
        assert_eq!(pq.sub_dim, 4);

        let test_vec = samples[0].clone();
        let codes = pq.quantize(&test_vec);
        assert_eq!(codes.len(), 4);

        let recon = pq.dequantize(&codes);
        assert_eq!(recon.len(), 16);

        let err = pq.quantization_error(&test_vec);
        assert!(err < 0.5, "PQ error should be reasonable, got {}", err); // loose for tiny k

        // Also test scalar fallback path still works
        let q = quantize(&test_vec);
        let dq = dequantize(&q);
        assert_eq!(dq.len(), 16);
    }

    #[test]
    fn test_pq_error_helper() {
        let v = vec![0.5f32; 8];
        let pq = ProductQuantizer::train(&[v.clone()], 2, 4);
        let err = pq.quantization_error(&v);
        assert!(err >= 0.0);
    }

    // Phase 9: property style for roundtrip
    #[test]
    fn test_pq_roundtrip_property() {
        for i in 0..10 {
            let mut v = vec![0.0f32; 16];
            for (j, x) in v.iter_mut().enumerate() {
                *x = ((i + j) as f32 / 10.0) - 0.5;
            }
            let norm: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt().max(1e-6);
            for x in &mut v { *x /= norm; }
            let pq = ProductQuantizer::train(&[v.clone()], 4, 8);
            let codes = pq.quantize(&v);
            let back = pq.dequantize(&codes);
            let err = pq.quantization_error(&v);
            assert!(err < 0.5, "roundtrip err too high");
            assert_eq!(back.len(), v.len());
        }
    }
}

/// Real Product Quantization (PQ) with k-means (Phase 8).
///
/// Product Quantization splits a high-dim vector (e.g. 384) into `m` subvectors,
/// and replaces each subvector with the index of the nearest centroid from a
/// learned codebook of size `k` (typically 256 for 1-byte codes).
///
/// This gives much better compression than per-dimension scalar quant (M bytes
/// vs D bytes) while keeping reasonable fidelity.
///
/// Training:
///   let pq = ProductQuantizer::train(&sample_embeddings, m=8, k=256);
///   let codes: Vec<u8> = pq.quantize(&vec);
///   let approx = pq.dequantize(&codes);
///
/// The implementation uses a simple Lloyd k-means per subspace.
/// Centroids are stored and used at runtime. For production you would
/// persist the trained PQ alongside the collection.
#[derive(Clone, Debug)]
pub struct ProductQuantizer {
    /// Number of subquantizers (subspaces). dim must be divisible by m.
    pub m: usize,
    /// Number of centroids per subquantizer (256 => u8 code).
    pub k: usize,
    /// sub_dim = dim / m
    sub_dim: usize,
    /// codebooks[m][k][sub_dim]
    codebooks: Vec<Vec<Vec<f32>>>,
}

impl ProductQuantizer {
    /// Create an untrained (empty) PQ. Call `train` before use.
    pub fn new() -> Self {
        Self {
            m: 0,
            k: 0,
            sub_dim: 0,
            codebooks: vec![],
        }
    }

    /// Train a PQ on sample vectors (should be representative normalized embeddings).
    /// m = number of subquantizers (e.g. 8 for 384d -> 48d subs)
    /// k = number of centroids (256 recommended)
    pub fn train(samples: &[Vec<f32>], m: usize, k: usize) -> Self {
        if samples.is_empty() {
            return Self::new();
        }
        let dim = samples[0].len();
        if dim % m != 0 || m == 0 {
            // fallback to 1 sub (whole vector) or panic in real; here we adjust
            let m = if dim > 0 { 1 } else { 0 };
            let sub_dim = dim;
            // degenerate
            let centroids = kmeans_subspace(samples, k, sub_dim, 12);
            return Self {
                m,
                k,
                sub_dim,
                codebooks: vec![centroids],
            };
        }

        let sub_dim = dim / m;

        let mut codebooks = Vec::with_capacity(m);

        for i in 0..m {
            // Extract i-th subspace for all samples
            let sub_samples: Vec<Vec<f32>> = samples
                .iter()
                .map(|v| v[i * sub_dim..(i + 1) * sub_dim].to_vec())
                .collect();

            let centroids = kmeans_subspace(&sub_samples, k, sub_dim, 12);
            codebooks.push(centroids);
        }

        Self {
            m,
            k,
            sub_dim,
            codebooks,
        }
    }

    /// Returns true if codebooks have been trained.
    pub fn is_trained(&self) -> bool {
        !self.codebooks.is_empty() && self.m > 0
    }

    /// Quantize a vector to M codes (one byte per subvector).
    /// Returns vec of length `m`.
    pub fn quantize(&self, vec: &[f32]) -> Vec<u8> {
        if !self.is_trained() || vec.len() != self.m * self.sub_dim {
            // fall back to scalar for safety in mixed use
            return quantize(vec);
        }

        let mut codes = Vec::with_capacity(self.m);
        for i in 0..self.m {
            let sub = &vec[i * self.sub_dim..(i + 1) * self.sub_dim];
            let cb = &self.codebooks[i];
            let idx = nearest_centroid(sub, cb);
            codes.push(idx as u8);
        }
        codes
    }

    /// Approximate reconstruction from PQ codes.
    pub fn dequantize(&self, codes: &[u8]) -> Vec<f32> {
        if !self.is_trained() || codes.len() != self.m {
            return dequantize(codes);
        }

        let mut out = Vec::with_capacity(self.m * self.sub_dim);
        for (i, &c) in codes.iter().enumerate() {
            let idx = (c as usize).min(self.k - 1);
            if let Some(cent) = self.codebooks.get(i).and_then(|cb| cb.get(idx)) {
                out.extend_from_slice(cent);
            }
        }
        out
    }

    /// Convenience: quantize + store as single byte vec (the codes themselves).
    pub fn quantize_to_bytes(&self, vec: &[f32]) -> Vec<u8> {
        self.quantize(vec)
    }

    /// Compute approx quantization error using current codebooks (RMS).
    pub fn quantization_error(&self, orig: &[f32]) -> f64 {
        if !self.is_trained() {
            return quantization_error(orig);
        }
        let q = self.quantize(orig);
        let dq = self.dequantize(&q);
        let mut sum_sq = 0.0f32;
        let n = orig.len().min(dq.len());
        for i in 0..n {
            let d = orig[i] - dq[i];
            sum_sq += d * d;
        }
        if n == 0 { 0.0 } else { (sum_sq / n as f32).sqrt() as f64 }
    }
}

// --- Internal k-means helpers (Lloyd algorithm) ---

fn kmeans_subspace(samples: &[Vec<f32>], k: usize, dim: usize, max_iters: usize) -> Vec<Vec<f32>> {
    if samples.is_empty() || k == 0 {
        return vec![vec![0.0; dim]; k.max(1)];
    }
    let k = k.min(samples.len());

    // Initialize: pick first k distinct samples (or random-ish)
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    for i in 0..k {
        let idx = (i * 7) % samples.len(); // simple spread
        centroids.push(samples[idx].clone());
    }

    for _ in 0..max_iters {
        // Assignment
        let mut clusters: Vec<Vec<Vec<f32>>> = vec![vec![]; k];
        for s in samples {
            let cidx = nearest_centroid(s, &centroids);
            clusters[cidx].push(s.clone());
        }

        let mut changed = false;
        for (ci, cluster) in clusters.iter().enumerate() {
            if cluster.is_empty() {
                continue;
            }
            let new_c = mean_vector(cluster, dim);
            if euclid_dist(&centroids[ci], &new_c) > 1e-6 {
                centroids[ci] = new_c;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    centroids
}

#[inline]
fn nearest_centroid(vec: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0;
    let mut best_d = f32::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let d = euclid_dist(vec, c);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

#[inline]
fn euclid_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

fn mean_vector(vectors: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; dim];
    if vectors.is_empty() {
        return mean;
    }
    for v in vectors {
        for (i, &x) in v.iter().enumerate().take(dim) {
            mean[i] += x;
        }
    }
    let n = vectors.len() as f32;
    for x in &mut mean {
        *x /= n;
    }
    mean
}

/// Default trained Product Quantizer for storage (Phase 8 polish).
/// Uses M=8 subquantizers, K=256 centroids on 384-dim vectors.
/// Trained on synthetic normalized data at first use.
/// This enables much smaller storage (8 bytes per vector vs 384).
pub fn default_product_quantizer() -> ProductQuantizer {
    static DEFAULT_PQ: std::sync::OnceLock<ProductQuantizer> = std::sync::OnceLock::new();
    DEFAULT_PQ.get_or_init(|| {
        let dim = 384;
        let m = 8;
        let k = 256;
        let mut samples = Vec::with_capacity(1024);
        let mut seed: u64 = 0x1234567890abcdef;
        for _ in 0..1024 {
            let mut v = vec![0.0f32; dim];
            for x in &mut v {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *x = ((seed >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            }
            // L2 normalize
            let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            samples.push(v);
        }
        ProductQuantizer::train(&samples, m, k)
    }).clone()
}

// PQ test moved to avoid duplicate mod
// #[cfg(test)]
// mod tests { ... }
