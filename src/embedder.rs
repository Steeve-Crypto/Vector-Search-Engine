//! Embedder module (Phase 1)
//!
//! Loads the `sentence-transformers/all-MiniLM-L6-v2` model using ONNX Runtime (ort)
//! and the Hugging Face tokenizer.
//!
//! Features:
//! - Automatic download of model + tokenizer on first use (or via CLI `download-model`)
//! - Proper mean-pooling + L2 normalization (matches sentence-transformers behavior)
//! - Single + batch embedding
//! - Lazy singleton session (loaded once per process)
//! - Thread-safe
//!
//! Model files are stored under:
//!   models/all-MiniLM-L6-v2/onnx/model.onnx
//!   models/all-MiniLM-L6-v2/tokenizer.json

use ndarray::{Array1, Array2, Array3};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use thiserror::Error;
use tokenizers::Tokenizer;
use tracing::{info, warn};

/// Embedding dimension for all-MiniLM-L6-v2.
pub const EMBED_DIM: usize = 384;

/// Default model directory.
const DEFAULT_MODEL_DIR: &str = "models/all-MiniLM-L6-v2";

/// Expected relative paths inside the model dir.
const ONNX_SUBPATH: &str = "onnx/model.onnx";
const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// Hugging Face download URLs (resolve/main for the raw file).
const HF_MODEL_URL: &str =
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx";
const HF_TOKENIZER_URL: &str =
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json";

/// Errors specific to the embedder.
#[derive(Error, Debug)]
pub enum EmbedderError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP download error: {0}")]
    Download(String),

    #[error("Failed to load tokenizer: {0}")]
    Tokenizer(String),

    #[error("ONNX Runtime error: {0}")]
    Ort(String),

    #[error("Shape or tensor error during inference: {0}")]
    Tensor(String),

    #[error("Model files not found and auto-download failed")]
    ModelNotAvailable,

    #[error("Input text cannot be empty")]
    EmptyText,
}

pub type Result<T> = std::result::Result<T, EmbedderError>;

/// The main embedder. Holds the loaded ONNX session and tokenizer.
/// 
/// Use the module-level `embed` / `embed_batch` helpers for convenience (they use the
/// process-wide singleton).
///
/// The session is behind a Mutex because `ort::Session::run` takes `&mut self`.
pub struct Embedder {
    session: std::sync::Mutex<Session>,
    tokenizer: Tokenizer,
}

impl Embedder {
    /// Load (or download) the model and create the embedder.
    /// This is relatively expensive — use the singleton helpers in practice.
    pub fn load() -> Result<Self> {
        let (model_path, tokenizer_path) = ensure_model_files()?;

        info!(?model_path, ?tokenizer_path, "loading embedding model + tokenizer");

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedderError::Tokenizer(e.to_string()))?;

        let session = Session::builder()
            .map_err(|e| EmbedderError::Ort(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| EmbedderError::Ort(e.to_string()))?
            .with_intra_threads(2)
            .map_err(|e| EmbedderError::Ort(e.to_string()))?
            .commit_from_file(&model_path)
            .map_err(|e| EmbedderError::Ort(e.to_string()))?;

        info!("embedding model loaded successfully (dim={})", EMBED_DIM);

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
        })
    }

    /// Embed a single text. Returns a normalized 384-dimensional vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if text.trim().is_empty() {
            return Err(EmbedderError::EmptyText);
        }
        let batch = self.embed_batch(&[text.to_owned()])?;
        Ok(batch.into_iter().next().unwrap())
    }

    /// Embed multiple texts. Returns a Vec of normalized 384-d vectors (same order).
    /// Efficiently runs a single batched ONNX inference.
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize all texts (add special tokens, let us pad ourselves)
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbedderError::Tokenizer(e.to_string()))?;

        let batch_size = texts.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        if max_len == 0 {
            // All empty after tokenization (shouldn't happen)
            return Ok(vec![vec![0.0; EMBED_DIM]; batch_size]);
        }

        // Build padded input arrays [batch, max_len]
        let mut input_ids_vec = vec![0i64; batch_size * max_len];
        let mut attention_mask_vec = vec![0i64; batch_size * max_len];

        for (b, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let seq_len = ids.len().min(max_len);

            for i in 0..seq_len {
                input_ids_vec[b * max_len + i] = ids[i] as i64;
                attention_mask_vec[b * max_len + i] = mask[i] as i64;
            }
            // remaining positions stay 0 (pad token + mask=0)
        }

        let input_ids = Array2::from_shape_vec((batch_size, max_len), input_ids_vec)
            .map_err(|e| EmbedderError::Tensor(e.to_string()))?;
        let attention_mask = Array2::from_shape_vec((batch_size, max_len), attention_mask_vec)
            .map_err(|e| EmbedderError::Tensor(e.to_string()))?;

        // Run inference
        // Wrap ndarrays into ort::value::Tensor (required by ort 2.0 inputs! macro)
        // Build tensors for ONNX (consume the arrays)
        let input_ids_data = input_ids.into_raw_vec();
        let attn_data = attention_mask.into_raw_vec();
        let type_ids_data = vec![0i64; batch_size * max_len]; // BERT-style models often require token_type_ids

        let input_ids_tensor = Tensor::<i64>::from_array(([batch_size, max_len], input_ids_data))
            .map_err(|e| EmbedderError::Ort(e.to_string()))?;
        let attention_mask_tensor =
            Tensor::<i64>::from_array(([batch_size, max_len], attn_data.clone()))
                .map_err(|e| EmbedderError::Ort(e.to_string()))?;
        let token_type_ids_tensor =
            Tensor::<i64>::from_array(([batch_size, max_len], type_ids_data))
                .map_err(|e| EmbedderError::Ort(e.to_string()))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| EmbedderError::Ort(e.to_string()))?;
        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ])
            .map_err(|e| EmbedderError::Ort(e.to_string()))?;

        // last_hidden_state is typically shape [batch, seq_len, 384]
        let last_hidden_value = &outputs["last_hidden_state"];
        let (shape, flat_data) = last_hidden_value
            .try_extract_tensor::<f32>()
            .map_err(|e| EmbedderError::Tensor(format!("extract last_hidden_state: {e}")))?;

        // Reconstruct Array3 [batch, seq, dim]
        let bsz = shape[0] as usize;
        let sqlen = shape[1] as usize;
        let hdim = shape[2] as usize;
        let hidden: Array3<f32> = Array3::from_shape_vec((bsz, sqlen, hdim), flat_data.to_vec())
            .map_err(|e| EmbedderError::Tensor(e.to_string()))?;

        // Mean pooling + L2 normalization
        // We still have the original vecs; rebuild a cheap mask view
        let mut results = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let mut sum = Array1::<f32>::zeros(EMBED_DIM);
            let mut count = 0f32;

            for s in 0..max_len {
                let mask_val = attn_data[b * max_len + s];
                if mask_val != 0 {
                    let token_vec = hidden.slice(ndarray::s![b, s, ..]);
                    sum = &sum + &token_vec;
                    count += 1.0;
                }
            }

            if count > 0.0 {
                sum /= count;
            }

            // L2 normalize
            let norm_sq: f32 = sum.iter().map(|&x| x * x).sum();
            let norm = norm_sq.sqrt();
            if norm > 1e-8 {
                sum /= norm;
            }

            results.push(sum.to_vec());
        }

        Ok(results)
    }
}

/// Global singleton for the embedder (lazy, thread-safe).
static EMBEDDER: OnceLock<Embedder> = OnceLock::new();

/// Returns a reference to the process-wide embedder (initializes on first call).
/// Will auto-download the model if the files are missing.
pub fn get_embedder() -> Result<&'static Embedder> {
    if let Some(e) = EMBEDDER.get() {
        return Ok(e);
    }

    let embedder = Embedder::load()?;
    // If another thread initialized it in the meantime, we just drop ours.
    let _ = EMBEDDER.set(embedder);
    Ok(EMBEDDER.get().expect("embedder was just inserted"))
}

/// Embed a single piece of text. This is the main public API.
/// 
/// Automatically downloads the model on first use if needed.
pub fn embed(text: &str) -> Result<Vec<f32>> {
    get_embedder()?.embed(text)
}

/// Batch version of `embed`.
pub fn embed_batch(texts: &[String]) -> Result<Vec<Vec<f32>>> {
    get_embedder()?.embed_batch(texts)
}

/// Ensure model + tokenizer exist on disk. Downloads if missing.
/// Returns the two paths (model, tokenizer).
fn ensure_model_files() -> Result<(PathBuf, PathBuf)> {
    let base_dir = PathBuf::from(DEFAULT_MODEL_DIR);
    let model_path = base_dir.join(ONNX_SUBPATH);
    let tokenizer_path = base_dir.join(TOKENIZER_FILENAME);

    if model_path.exists() && tokenizer_path.exists() {
        return Ok((model_path, tokenizer_path));
    }

    info!("embedding model files not found — starting download...");
    download_model_files(&base_dir, &model_path, &tokenizer_path)?;
    Ok((model_path, tokenizer_path))
}

/// Download both files. Creates directories as needed.
fn download_model_files(_base_dir: &Path, model_path: &Path, tokenizer_path: &Path) -> Result<()> {
    std::fs::create_dir_all(model_path.parent().unwrap())?;

    // Download model (large)
    download_to_file(HF_MODEL_URL, model_path, "model.onnx (~86 MB)")?;

    // Download tokenizer (small)
    download_to_file(HF_TOKENIZER_URL, tokenizer_path, "tokenizer.json")?;

    info!("model download complete");
    Ok(())
}

/// Simple streaming download using ureq (respects the vendored / rustls setup).
fn download_to_file(url: &str, dest: &Path, label: &str) -> Result<()> {
    info!("downloading {} from {}", label, url);

    let response = ureq::get(url)
        .call()
        .map_err(|e| EmbedderError::Download(format!("request to {url}: {e}")))?;

    if response.status() != 200 {
        return Err(EmbedderError::Download(format!(
            "bad status {} for {}",
            response.status(),
            url
        )));
    }

    let mut reader = response.into_body().into_reader();
    let mut file = std::fs::File::create(dest)?;

    std::io::copy(&mut reader, &mut file)?;

    info!("saved {} to {}", label, dest.display());
    Ok(())
}

/// Force (re)download of the model files.
/// Useful for the CLI `download-model --force` command.
pub fn download_model_if_needed(force: bool) -> Result<()> {
    let base_dir = PathBuf::from(DEFAULT_MODEL_DIR);
    let model_path = base_dir.join(ONNX_SUBPATH);
    let tokenizer_path = base_dir.join(TOKENIZER_FILENAME);

    if force {
        if model_path.exists() {
            let _ = std::fs::remove_file(&model_path);
        }
        if tokenizer_path.exists() {
            let _ = std::fs::remove_file(&tokenizer_path);
        }
        warn!("forced re-download of embedding model");
    }

    if model_path.exists() && tokenizer_path.exists() && !force {
        info!("model files already present");
        return Ok(());
    }

    download_model_files(&base_dir, &model_path, &tokenizer_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_dimension_and_normalization() {
        // This will trigger download + load on first run in this test suite.
        let vec = embed("This is a test sentence for the vector search engine.")
            .expect("embed should succeed (model will be downloaded if needed)");

        assert_eq!(
            vec.len(),
            EMBED_DIM,
            "embedding must be exactly {} dimensions",
            EMBED_DIM
        );

        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "L2 norm should be ~1.0 after normalization, got {:.8}",
            norm
        );
    }

    #[test]
    fn test_embed_batch_consistency() {
        let texts = vec![
            "Rust is fast and safe.".to_string(),
            "Python is great for data science.".to_string(),
        ];

        let embs = embed_batch(&texts).expect("batch embed failed");
        assert_eq!(embs.len(), 2);
        assert_eq!(embs[0].len(), EMBED_DIM);
        assert_eq!(embs[1].len(), EMBED_DIM);

        // Both must be normalized
        for emb in &embs {
            let n: f32 = emb.iter().map(|&x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "batch item norm was {n}");
        }

        // Cosine between two different sentences should be < 1.0 (they are not identical)
        let dot: f32 = embs[0]
            .iter()
            .zip(embs[1].iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(dot < 0.999, "distinct sentences should not have cosine ~1.0");
    }

    #[test]
    fn test_empty_input_error() {
        let res = embed("");
        assert!(matches!(res, Err(EmbedderError::EmptyText)));
    }
}
