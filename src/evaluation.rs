use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, layer_norm, ops};
use hf_hub::api::sync::Api;
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExactMetrics {
    pub solved_pct: f64,
    pub solved: usize,
    pub total: usize,
    pub cell_pct: f64,
    pub valid_pct: f64,
}

#[must_use]
pub fn evaluate_exact(hyps: &[String], refs: &[String]) -> ExactMetrics {
    let total = hyps.len();
    if total == 0 {
        return ExactMetrics {
            solved_pct: 0.0,
            solved: 0,
            total: 0,
            cell_pct: 0.0,
            valid_pct: 0.0,
        };
    }

    let mut solved = 0usize;
    let mut valid = 0usize;
    let mut cell_sum = 0.0f64;

    for (h, r) in hyps.iter().zip(refs.iter()) {
        if h == r {
            solved += 1;
        }
        let hc: Vec<char> = h.chars().filter(|&c| c != '|').collect();
        let rc: Vec<char> = r.chars().filter(|&c| c != '|').collect();
        let denom = hc.len().min(rc.len()).max(1);
        let matches = hc.iter().zip(rc.iter()).filter(|(a, b)| a == b).count();
        cell_sum += matches as f64 / denom as f64;

        if is_valid_sudoku(&hc) {
            valid += 1;
        }
    }

    ExactMetrics {
        solved_pct: 100.0 * solved as f64 / total as f64,
        solved,
        total,
        cell_pct: 100.0 * cell_sum / total as f64,
        valid_pct: 100.0 * valid as f64 / total as f64,
    }
}

fn is_valid_sudoku(grid: &[char]) -> bool {
    if grid.len() != 81 || !grid.iter().all(|c| ('1'..='9').contains(c)) {
        return false;
    }
    let is_perm = |slice: &[char]| -> bool {
        let mut seen = [false; 9];
        for &c in slice {
            let idx = (c as usize) - ('1' as usize);
            if seen[idx] {
                return false;
            }
            seen[idx] = true;
        }
        true
    };

    for i in 0..9 {
        if !is_perm(&grid[i * 9..(i + 1) * 9]) {
            return false;
        }
        let col: Vec<char> = (0..9).map(|r| grid[r * 9 + i]).collect();
        if !is_perm(&col) {
            return false;
        }
    }
    for br in (0..9).step_by(3) {
        for bc in (0..9).step_by(3) {
            let mut box_cells = Vec::with_capacity(9);
            for r in br..br + 3 {
                for c in bc..bc + 3 {
                    box_cells.push(grid[r * 9 + c]);
                }
            }
            if !is_perm(&box_cells) {
                return false;
            }
        }
    }
    true
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RougeMetrics {
    pub rouge1: f64,
    pub rouge2: f64,
    pub rouge_l: f64,
    pub mean_words: f64,
}

fn rouge_tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lc in ch.to_lowercase() {
                cur.push(lc);
            }
        } else if !cur.is_empty() {
            tokens.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}

fn fmeasure(overlap: usize, hyp_len: usize, ref_len: usize) -> f64 {
    if hyp_len == 0 || ref_len == 0 || overlap == 0 {
        return 0.0;
    }
    let prec = overlap as f64 / hyp_len as f64;
    let rec = overlap as f64 / ref_len as f64;
    2.0 * prec * rec / (prec + rec)
}

fn ngram_f1(hyp: &[String], r: &[String], n: usize) -> f64 {
    if hyp.len() < n || r.len() < n {
        return 0.0;
    }
    let mut ref_counts: HashMap<&[String], usize> = HashMap::new();
    for w in r.windows(n) {
        *ref_counts.entry(w).or_insert(0) += 1;
    }
    let mut overlap = 0usize;
    for w in hyp.windows(n) {
        if let Some(c) = ref_counts.get_mut(w)
            && *c > 0
        {
            *c -= 1;
            overlap += 1;
        }
    }
    fmeasure(overlap, hyp.len() - n + 1, r.len() - n + 1)
}

fn lcs_len(a: &[String], b: &[String]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let mut dp = vec![0usize; b.len() + 1];
    for x in a {
        let mut prev = 0usize;
        for (j, y) in b.iter().enumerate() {
            let tmp = dp[j + 1];
            if x == y {
                dp[j + 1] = prev + 1;
            } else if dp[j] > dp[j + 1] {
                dp[j + 1] = dp[j];
            }
            prev = tmp;
        }
    }
    dp[b.len()]
}

#[must_use]
pub fn evaluate_rouge(hyps: &[String], refs: &[String]) -> RougeMetrics {
    let n = hyps.len().max(1) as f64;
    let mut r1 = 0.0;
    let mut r2 = 0.0;
    let mut rl = 0.0;
    let mut words = 0usize;

    for (h, r) in hyps.iter().zip(refs.iter()) {
        words += h.split_whitespace().count();
        let ht = rouge_tokenize(h);
        let rt = rouge_tokenize(r);
        r1 += ngram_f1(&ht, &rt, 1);
        r2 += ngram_f1(&ht, &rt, 2);
        rl += fmeasure(lcs_len(&ht, &rt), ht.len(), rt.len());
    }

    RougeMetrics {
        rouge1: 100.0 * r1 / n,
        rouge2: 100.0 * r2 / n,
        rouge_l: 100.0 * rl / n,
        mean_words: words as f64 / n,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BleuMetrics {
    pub bleu_cased: f64,
    pub bleu_lowercase: f64,
    pub chrf: f64,
    pub len_ratio: f64,
}

fn tokenize_13a(s: &str, lowercase: bool) -> Vec<String> {
    let text = if lowercase {
        s.to_lowercase()
    } else {
        s.to_string()
    };
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else if ch.is_ascii_punctuation() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            out.push(ch.to_string());
        } else {
            cur.push(ch);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn corpus_bleu(hyps: &[String], refs: &[String], lowercase: bool) -> f64 {
    let mut clipped = [0usize; 4];
    let mut total = [0usize; 4];
    let mut hyp_len = 0usize;
    let mut ref_len = 0usize;

    for (h, r) in hyps.iter().zip(refs.iter()) {
        let ht = tokenize_13a(h, lowercase);
        let rt = tokenize_13a(r, lowercase);
        hyp_len += ht.len();
        ref_len += rt.len();

        for n in 1..=4 {
            if ht.len() < n {
                continue;
            }
            total[n - 1] += ht.len() - n + 1;
            let mut ref_counts: HashMap<&[String], usize> = HashMap::new();
            for w in rt.windows(n) {
                *ref_counts.entry(w).or_insert(0) += 1;
            }
            for w in ht.windows(n) {
                if let Some(c) = ref_counts.get_mut(w)
                    && *c > 0
                {
                    *c -= 1;
                    clipped[n - 1] += 1;
                }
            }
        }
    }

    if hyp_len == 0 {
        return 0.0;
    }
    let bp = if hyp_len > ref_len {
        1.0
    } else {
        (1.0 - (ref_len as f64 / hyp_len as f64)).exp()
    };

    let mut log_sum = 0.0;
    for i in 0..4 {
        if clipped[i] == 0 || total[i] == 0 {
            return 0.0;
        }
        log_sum += (clipped[i] as f64 / total[i] as f64).ln();
    }
    100.0 * bp * (log_sum / 4.0).exp()
}

fn corpus_chrf(hyps: &[String], refs: &[String]) -> f64 {
    let beta_sq = 4.0f64;
    let mut f_sum = 0.0;
    let mut n_orders = 0usize;
    for n in 1..=6 {
        let mut overlap = 0usize;
        let mut hyp_tot = 0usize;
        let mut ref_tot = 0usize;
        for (h, r) in hyps.iter().zip(refs.iter()) {
            let hc: Vec<char> = h.chars().filter(|c| !c.is_whitespace()).collect();
            let rc: Vec<char> = r.chars().filter(|c| !c.is_whitespace()).collect();
            if hc.len() >= n {
                hyp_tot += hc.len() - n + 1;
            }
            if rc.len() >= n {
                ref_tot += rc.len() - n + 1;
            }
            if hc.len() >= n && rc.len() >= n {
                let mut ref_counts: HashMap<&[char], usize> = HashMap::new();
                for w in rc.windows(n) {
                    *ref_counts.entry(w).or_insert(0) += 1;
                }
                for w in hc.windows(n) {
                    if let Some(c) = ref_counts.get_mut(w)
                        && *c > 0
                    {
                        *c -= 1;
                        overlap += 1;
                    }
                }
            }
        }
        if hyp_tot > 0 && ref_tot > 0 && overlap > 0 {
            let p = overlap as f64 / hyp_tot as f64;
            let rec = overlap as f64 / ref_tot as f64;
            f_sum += (1.0 + beta_sq) * p * rec / (beta_sq * p + rec);
        }
        n_orders += 1;
    }
    100.0 * f_sum / n_orders as f64
}

#[must_use]
pub fn evaluate_bleu(hyps: &[String], refs: &[String]) -> BleuMetrics {
    let hyp_words: usize = hyps.iter().map(|h| h.split_whitespace().count()).sum();
    let ref_words: usize = refs.iter().map(|r| r.split_whitespace().count()).sum();
    BleuMetrics {
        bleu_cased: corpus_bleu(hyps, refs, false),
        bleu_lowercase: corpus_bleu(hyps, refs, true),
        chrf: corpus_chrf(hyps, refs),
        len_ratio: hyp_words as f64 / ref_words.max(1) as f64,
    }
}

#[must_use]
pub fn token_entropy(ids: &[u32]) -> f64 {
    if ids.is_empty() {
        return 0.0;
    }
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for &id in ids {
        *counts.entry(id).or_insert(0) += 1;
    }
    let total = ids.len() as f64;
    let mut ent = 0.0;
    for &c in counts.values() {
        let p = c as f64 / total;
        ent -= p * p.ln();
    }
    ent
}

#[derive(Debug, Deserialize)]
struct Gpt2ConfigJson {
    vocab_size: usize,
    n_positions: usize,
    n_embd: usize,
    n_layer: usize,
    n_head: usize,
    #[serde(default = "default_ln_eps")]
    layer_norm_epsilon: f64,
}

fn default_ln_eps() -> f64 {
    1e-5
}

struct Gpt2Block {
    ln_1: LayerNorm,
    c_attn: Linear,
    c_proj: Linear,
    ln_2: LayerNorm,
    mlp_fc: Linear,
    mlp_proj: Linear,
    n_head: usize,
    n_embd: usize,
}

impl Gpt2Block {
    fn load(cfg: &Gpt2ConfigJson, vb: VarBuilder) -> candle_core::Result<Self> {
        let eps = cfg.layer_norm_epsilon;
        Ok(Self {
            ln_1: layer_norm(cfg.n_embd, eps, vb.pp("ln_1"))?,
            c_attn: conv1d_as_linear(cfg.n_embd, 3 * cfg.n_embd, vb.pp("attn").pp("c_attn"))?,
            c_proj: conv1d_as_linear(cfg.n_embd, cfg.n_embd, vb.pp("attn").pp("c_proj"))?,
            ln_2: layer_norm(cfg.n_embd, eps, vb.pp("ln_2"))?,
            mlp_fc: conv1d_as_linear(cfg.n_embd, 4 * cfg.n_embd, vb.pp("mlp").pp("c_fc"))?,
            mlp_proj: conv1d_as_linear(4 * cfg.n_embd, cfg.n_embd, vb.pp("mlp").pp("c_proj"))?,
            n_head: cfg.n_head,
            n_embd: cfg.n_embd,
        })
    }

    fn forward(&self, x: &Tensor, causal_mask: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let hd = self.n_embd / self.n_head;
        let h = self.ln_1.forward(x)?;
        let qkv = self.c_attn.forward(&h)?;
        let q = qkv
            .narrow(D::Minus1, 0, c)?
            .reshape((b, t, self.n_head, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = qkv
            .narrow(D::Minus1, c, c)?
            .reshape((b, t, self.n_head, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = qkv
            .narrow(D::Minus1, 2 * c, c)?
            .reshape((b, t, self.n_head, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let scale = 1.0 / (hd as f64).sqrt();
        let scores =
            (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?.broadcast_add(causal_mask)?;
        let probs = ops::softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(v.dtype())?;
        let attn = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, c))?;
        let x = (x + self.c_proj.forward(&attn)?)?;

        let m = self.ln_2.forward(&x)?;
        let m = self.mlp_fc.forward(&m)?.gelu()?;
        let m = self.mlp_proj.forward(&m)?;
        x + m
    }
}

fn conv1d_as_linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> candle_core::Result<Linear> {
    // HF GPT-2 stores Conv1D weights as [in_dim, out_dim] instead of [out_dim, in_dim].
    let w = vb.get((in_dim, out_dim), "weight")?.t()?.contiguous()?;
    let b = vb.get(out_dim, "bias")?;
    Ok(Linear::new(w, Some(b)))
}

pub struct Gpt2Scorer {
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Gpt2Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
    tokenizer: Tokenizer,
    eos_token_id: u32,
    n_positions: usize,
    device: Device,
}

impl Gpt2Scorer {
    pub fn load(model_id: &str, device: Device, dtype: DType) -> Result<Self> {
        let local_dir = Path::new(model_id);
        let (cfg_path, tok_path, weights_path) = if local_dir.is_dir() {
            (
                local_dir.join("config.json"),
                local_dir.join("tokenizer.json"),
                local_dir.join("model.safetensors"),
            )
        } else {
            let api = Api::new().context("Failed to initialize hf-hub API")?;
            let repo = api.model(model_id.to_string());
            (
                repo.get("config.json")?,
                repo.get("tokenizer.json")?,
                repo.get("model.safetensors")?,
            )
        };

        let cfg: Gpt2ConfigJson = serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)?;
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("Failed to load GPT-2 tokenizer: {e}"))?;
        let eos_token_id = tokenizer.token_to_id("<|endoftext|>").unwrap_or(50_256);

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)? };
        let wte = embedding(cfg.vocab_size, cfg.n_embd, vb.pp("wte"))?;
        let wpe = embedding(cfg.n_positions, cfg.n_embd, vb.pp("wpe"))?;
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            blocks.push(Gpt2Block::load(&cfg, vb.pp("h").pp(i))?);
        }
        let ln_f = layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln_f"))?;
        let lm_head = Linear::new(wte.embeddings().clone(), None);

        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
            tokenizer,
            eos_token_id,
            n_positions: cfg.n_positions,
            device,
        })
    }

    pub fn mean_ppl(&self, texts: &[String], max_length: Option<usize>) -> Result<f64> {
        let max_len = max_length.unwrap_or(self.n_positions).min(self.n_positions);
        let mut total_nll = 0.0f64;
        let mut total_tok = 0usize;

        for text in texts {
            let enc = self
                .tokenizer
                .encode(text.as_str(), false)
                .map_err(|e| anyhow::anyhow!("GPT-2 encode failed: {e}"))?;
            let mut ids = enc.get_ids().to_vec();
            if ids.len() > max_len {
                ids.truncate(max_len);
            }
            if ids.len() < 2 {
                continue;
            }
            let t = ids.len();
            let input_ids = Tensor::from_vec(ids.clone(), (1, t), &self.device)?;
            let pos = Tensor::arange(0u32, t as u32, &self.device)?;
            let mut h = self
                .wte
                .forward(&input_ids)?
                .broadcast_add(&self.wpe.forward(&pos)?)?;

            let mut mask_vec = vec![0.0f32; t * t];
            for i in 0..t {
                for j in (i + 1)..t {
                    mask_vec[i * t + j] = f32::NEG_INFINITY;
                }
            }
            let mask =
                Tensor::from_vec(mask_vec, (1, 1, t, t), &self.device)?.to_dtype(h.dtype())?;
            for blk in &self.blocks {
                h = blk.forward(&h, &mask)?;
            }
            let h = self.ln_f.forward(&h)?;
            let logits = self.lm_head.forward(&h)?.i((0, ..t - 1, ..))?;
            let log_probs = ops::log_softmax(&logits.to_dtype(DType::F32)?, D::Minus1)?;
            let lp_rows = log_probs.to_vec2::<f32>()?;

            for (pos_idx, &target) in ids[1..].iter().enumerate() {
                total_nll -= f64::from(lp_rows[pos_idx][target as usize]);
                total_tok += 1;
                if target == self.eos_token_id {
                    break;
                }
            }
        }

        if total_tok == 0 {
            bail!("No valid tokens to score for perplexity");
        }
        Ok((total_nll / total_tok as f64).exp())
    }
}
