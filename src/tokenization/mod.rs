pub mod bpe;
pub mod hf;
pub mod sudoku;

use std::path::Path;

use anyhow::{Result, bail};

use crate::config::PreTokenizerConfig;

pub trait PreTokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn eos_token_id(&self) -> Option<u32>;
    fn pad_token_id(&self) -> Option<u32>;

    fn decode_batch(
        &self,
        ids_batch: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        ids_batch
            .iter()
            .map(|ids| self.decode(ids, skip_special_tokens))
            .collect()
    }
}

pub fn build_pre_tokenizer(
    cfg: &PreTokenizerConfig,
    checkpoint_dir: Option<&Path>,
) -> Result<Box<dyn PreTokenizer>> {
    match cfg.name.as_str() {
        "sudoku" => Ok(Box::new(sudoku::SudokuTokenizer::new())),
        "bpe" => {
            let vocab_rel = cfg
                .vocab_file_path
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("bpe tokenizer requires vocab_file_path"))?;
            let merges_rel = cfg
                .merges_file_path
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("bpe tokenizer requires merges_file_path"))?;
            let resolve = |p: &str| {
                let path = Path::new(p);
                if path.is_absolute() || path.exists() {
                    path.to_path_buf()
                } else if let Some(dir) = checkpoint_dir {
                    dir.join(path)
                } else {
                    path.to_path_buf()
                }
            };
            Ok(Box::new(bpe::BPETokenizer::from_files(
                &resolve(vocab_rel),
                &resolve(merges_rel),
                &cfg.special_tokens,
            )?))
        }
        "hf" => {
            let model_id = cfg.model_id.as_deref().unwrap_or("gpt2");
            let eos = cfg.special_tokens.get("eos_token").map(String::as_str);
            let pad = cfg.special_tokens.get("pad_token").map(String::as_str);
            Ok(Box::new(hf::HFTokenizer::from_pretrained(
                model_id, pad, eos,
            )?))
        }
        other => bail!("Unknown pre_tokenizer: {other}"),
    }
}
