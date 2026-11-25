use std::path::Path;

use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use tokenizers::{AddedToken, Tokenizer};

use super::PreTokenizer;

pub struct HFTokenizer {
    tokenizer: Tokenizer,
    vocab_size: usize,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
}

impl HFTokenizer {
    pub fn from_pretrained(
        model_id: &str,
        pad_token: Option<&str>,
        eos_token: Option<&str>,
    ) -> Result<Self> {
        let local_json = Path::new(model_id).join("tokenizer.json");
        let tokenizer_path = if Path::new(model_id).is_file() {
            Path::new(model_id).to_path_buf()
        } else if local_json.is_file() {
            local_json
        } else {
            let api = Api::new().context("Failed to initialize hf-hub API")?;
            let repo = api.model(model_id.to_string());
            repo.get("tokenizer.json")
                .with_context(|| format!("Failed to download tokenizer.json for {model_id}"))?
        };

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow::anyhow!("Failed to load tokenizer {}: {e}", tokenizer_path.display())
        })?;

        let eos_str = eos_token.unwrap_or("<|endoftext|>");
        let eos_token_id = tokenizer.token_to_id(eos_str).or_else(|| {
            tokenizer.add_special_tokens(&[AddedToken::from(eos_str, true)]);
            tokenizer.token_to_id(eos_str)
        });

        let pad_token_id = if let Some(pad_str) = pad_token {
            if let Some(id) = tokenizer.token_to_id(pad_str) {
                Some(id)
            } else {
                tokenizer.add_special_tokens(&[AddedToken::from(pad_str, true)]);
                tokenizer.token_to_id(pad_str)
            }
        } else {
            eos_token_id
        };

        let vocab_size = tokenizer.get_vocab_size(false);
        Ok(Self {
            tokenizer,
            vocab_size,
            eos_token_id,
            pad_token_id,
        })
    }
}

impl PreTokenizer for HFTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("HF encode failed: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("HF decode failed: {e}"))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }
}
