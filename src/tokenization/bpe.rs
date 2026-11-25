use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result, bail};
use fancy_regex::Regex;
use serde_pickle::{DeOptions, HashableValue, Value};

use super::PreTokenizer;

const GPT2_PATTERN: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub struct BPETokenizer {
    vocab: HashMap<u32, Vec<u8>>,
    reverse_vocab: HashMap<Vec<u8>, u32>,
    merges: HashMap<(Vec<u8>, Vec<u8>), usize>,
    special_tokens: Vec<String>,
    special_token_ids: HashSet<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
    regex: Regex,
}

impl BPETokenizer {
    pub fn from_files(
        vocab_path: &Path,
        merges_path: &Path,
        special_tokens: &HashMap<String, String>,
    ) -> Result<Self> {
        let vocab = load_vocab_pickle(vocab_path)?;
        let merges_vec = load_merges_pickle(merges_path)?;
        Self::new(vocab, merges_vec, special_tokens)
    }

    pub fn new(
        vocab: HashMap<u32, Vec<u8>>,
        merges_vec: Vec<(Vec<u8>, Vec<u8>)>,
        special_tokens_map: &HashMap<String, String>,
    ) -> Result<Self> {
        let reverse_vocab: HashMap<Vec<u8>, u32> = vocab
            .iter()
            .map(|(&id, bytes)| (bytes.clone(), id))
            .collect();
        let merges: HashMap<(Vec<u8>, Vec<u8>), usize> = merges_vec
            .into_iter()
            .enumerate()
            .map(|(idx, pair)| (pair, idx))
            .collect();

        let mut special_tokens: Vec<String> = special_tokens_map.values().cloned().collect();
        special_tokens.sort_by(|a, b| b.cmp(a));

        let mut token_to_idx = HashMap::new();
        for (idx, tok) in special_tokens_map.values().enumerate() {
            let id = reverse_vocab
                .get(tok.as_bytes())
                .copied()
                .unwrap_or(idx as u32);
            token_to_idx.insert(tok.clone(), id);
        }
        let special_token_ids: HashSet<u32> = token_to_idx.values().copied().collect();
        let eos_token_id = token_to_idx.get("<|endoftext|>").copied();
        let pad_token_id = token_to_idx.get("<|pad|>").copied();

        let regex = Regex::new(GPT2_PATTERN).context("Invalid BPE regex")?;
        Ok(Self {
            vocab,
            reverse_vocab,
            merges,
            special_tokens,
            special_token_ids,
            eos_token_id,
            pad_token_id,
            regex,
        })
    }

    fn bpe_word(&self, word_bytes: &[u8]) -> Vec<Vec<u8>> {
        let mut parts: Vec<Vec<u8>> = word_bytes.iter().map(|&b| vec![b]).collect();
        loop {
            if parts.len() < 2 {
                break;
            }
            let mut best_idx = self.merges.len();
            let mut best_pair: Option<(Vec<u8>, Vec<u8>)> = None;
            for pair in parts.windows(2) {
                let key = (pair[0].clone(), pair[1].clone());
                if let Some(&rank) = self.merges.get(&key)
                    && rank < best_idx
                {
                    best_idx = rank;
                    best_pair = Some(key);
                }
            }
            let Some((left, right)) = best_pair else {
                break;
            };
            let mut merged = Vec::with_capacity(parts.len());
            let mut i = 0;
            while i < parts.len() {
                if i + 1 < parts.len() && parts[i] == left && parts[i + 1] == right {
                    let mut combined = left.clone();
                    combined.extend_from_slice(&right);
                    merged.push(combined);
                    i += 2;
                } else {
                    merged.push(parts[i].clone());
                    i += 1;
                }
            }
            parts = merged;
        }
        parts
    }
}

impl PreTokenizer for BPETokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut chunks: Vec<(&str, bool)> = Vec::new();
        if self.special_tokens.is_empty() {
            chunks.push((text, false));
        } else {
            let escaped: Vec<String> = self
                .special_tokens
                .iter()
                .map(|s| fancy_regex::escape(s).into_owned())
                .collect();
            let split_re = Regex::new(&escaped.join("|"))?;
            let mut last = 0;
            for mat in split_re.find_iter(text) {
                let m = mat?;
                if m.start() > last {
                    chunks.push((&text[last..m.start()], false));
                }
                chunks.push((&text[m.start()..m.end()], true));
                last = m.end();
            }
            if last < text.len() {
                chunks.push((&text[last..], false));
            }
        }

        let mut ids = Vec::new();
        for (chunk, is_special) in chunks {
            if chunk.is_empty() {
                continue;
            }
            if is_special {
                let id = self
                    .reverse_vocab
                    .get(chunk.as_bytes())
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("Special token not in vocab: {chunk}"))?;
                ids.push(id);
            } else {
                for mat in self.regex.find_iter(chunk) {
                    let m = mat?;
                    for token_bytes in self.bpe_word(m.as_str().as_bytes()) {
                        let id =
                            self.reverse_vocab
                                .get(&token_bytes)
                                .copied()
                                .ok_or_else(|| {
                                    anyhow::anyhow!("Token bytes not in vocab: {token_bytes:?}")
                                })?;
                        ids.push(id);
                    }
                }
            }
        }
        Ok(ids)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            if skip_special_tokens && self.special_token_ids.contains(&id) {
                continue;
            }
            if let Some(b) = self.vocab.get(&id) {
                bytes.extend_from_slice(b);
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }
}

fn value_to_bytes(v: &Value) -> Result<Vec<u8>> {
    match v {
        Value::Bytes(b) => Ok(b.clone()),
        Value::String(s) => Ok(s.as_bytes().to_vec()),
        other => bail!("Expected bytes or string in BPE pickle, got {other:?}"),
    }
}

fn hashable_to_u32(k: &HashableValue) -> Result<u32> {
    match k {
        HashableValue::I64(i) => Ok(*i as u32),
        other => bail!("Expected integer key in BPE vocab pickle, got {other:?}"),
    }
}

fn load_vocab_pickle(path: &Path) -> Result<HashMap<u32, Vec<u8>>> {
    let file =
        File::open(path).with_context(|| format!("Failed to open BPE vocab {}", path.display()))?;
    let val: Value = serde_pickle::from_reader(BufReader::new(file), DeOptions::default())
        .with_context(|| format!("Failed to parse BPE vocab pickle {}", path.display()))?;
    let Value::Dict(map) = val else {
        bail!("BPE vocab pickle is not a dict");
    };
    let mut out = HashMap::with_capacity(map.len());
    for (k, v) in map {
        out.insert(hashable_to_u32(&k)?, value_to_bytes(&v)?);
    }
    Ok(out)
}

fn load_merges_pickle(path: &Path) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open BPE merges {}", path.display()))?;
    let val: Value = serde_pickle::from_reader(BufReader::new(file), DeOptions::default())
        .with_context(|| format!("Failed to parse BPE merges pickle {}", path.display()))?;
    let Value::List(items) = val else {
        bail!("BPE merges pickle is not a list");
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let Value::Tuple(pair) = item else {
            bail!("BPE merge entry is not a tuple");
        };
        if pair.len() != 2 {
            bail!("BPE merge tuple length != 2");
        }
        out.push((value_to_bytes(&pair[0])?, value_to_bytes(&pair[1])?));
    }
    Ok(out)
}
