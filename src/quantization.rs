//! Simple scalar quantization for Phase 6 (memory optimization).
//!
//! Scalar quantization maps f32 [-1,1] (normalized embeddings) to u8 [0,255].
//! Reduces memory ~4x for storage, with small accuracy loss.
//! For HNSW, we can store quantized but dequantize for distance (or use quantized dist later).
//!
//! Not yet integrated into HNSW (would require custom dist or full rebuild).
//! Use for storage in sled or export.

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
    fn test_pq_basic() {
        let pq = ProductQuantizer::new();
        let v = vec![0.0; 4];
        let (q1, q2) = pq.quantize_pq(&v);
        let d = pq.dequantize_pq(&q1, &q2);
        assert_eq!(d.len(), 4);
    }
}

/// Basic Product Quantization (PQ) for Phase 6 (full PQ beyond scalar).
/// Splits vector into 2 subvectors, quantizes each with scalar for demo.
/// In real PQ, use k-means codebooks per subvector.
/// For 384 dim, subvec 192 each.
pub struct ProductQuantizer {
    // For demo, no learned codebooks, just scalar on subs.
}

impl ProductQuantizer {
    pub fn new() -> Self { Self {} }

    /// Quantize to two u8 vecs (one per subvector).
    pub fn quantize_pq(&self, vec: &[f32]) -> (Vec<u8>, Vec<u8>) {
        let mid = vec.len() / 2;
        let sub1 = quantize(&vec[..mid]);
        let sub2 = quantize(&vec[mid..]);
        (sub1, sub2)
    }

    pub fn dequantize_pq(&self, q1: &[u8], q2: &[u8]) -> Vec<f32> {
        let mut v = dequantize(q1);
        v.extend(dequantize(q2));
        v
    }
}

// PQ test moved to avoid duplicate mod
// #[cfg(test)]
// mod tests { ... }
