use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use candle_core::pickle::PthTensors;
use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;
use rand::Rng;
use rand::seq::index::sample as rand_sample_indices;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use serde_pickle::{DeOptions, Value};

use crate::config::{GenerationFileConfig, ModelEvalConfig};
use crate::models::next_concept::{GenerateOptions, NextConceptTransformer};
use crate::models::vqvae::Vqvae;
use crate::tokenization::{PreTokenizer, build_pre_tokenizer};

#[derive(Debug, Clone)]
pub struct TokenPickerConfig {
    pub frequency: f64,
    pub deficit: f64,
    pub strength: f64,
    pub window: usize,
    pub decay: f64,
    pub allowance: Option<Vec<f64>>,
}

impl TokenPickerConfig {
    #[must_use]
    pub fn from_settings(
        frequency: f64,
        deficit: f64,
        adjacency_spec: Option<&str>,
        allowance: Option<Vec<f64>>,
    ) -> Self {
        let mut strength = 0.0;
        let mut window = 0usize;
        let mut decay = 1.0;
        if let Some(spec) = adjacency_spec
            && !spec.is_empty()
        {
            let parts: Vec<f64> = spec
                .split(',')
                .filter_map(|s| s.trim().parse::<f64>().ok())
                .collect();
            if !parts.is_empty() {
                strength = parts[0];
                window = if parts.len() > 1 {
                    parts[1].max(0.0) as usize
                } else {
                    1
                };
                decay = if parts.len() > 2 { parts[2] } else { 1.0 };
            }
        }
        Self {
            frequency,
            deficit,
            strength,
            window,
            decay,
            allowance,
        }
    }
}

/// Frequency- and adjacency-penalized argmax over `[B, L, V]` logits.
pub fn pick_tokens(logits: &Tensor, cfg: &TokenPickerConfig) -> Result<Tensor> {
    if cfg.frequency <= 0.0 && cfg.window == 0 {
        return Ok(logits.argmax(D::Minus1)?);
    }
    let (batch, length, vocab) = logits.dims3()?;
    let rows = logits.to_dtype(DType::F64)?.to_vec3::<f64>()?;
    let mut out = Vec::with_capacity(batch * length);

    let allow_scaled: Option<Vec<f64>> = cfg
        .allowance
        .as_ref()
        .map(|a| a.iter().take(vocab).map(|&v| v * length as f64).collect());

    for batch_logits in rows {
        let mut counts = vec![0.0f64; vocab];
        let mut history: Vec<usize> = Vec::with_capacity(length);
        for (pos, mut step) in batch_logits.into_iter().enumerate() {
            if cfg.frequency > 0.0 {
                for (tok, score) in step.iter_mut().enumerate() {
                    let c = counts[tok];
                    if let Some(ref allow) = allow_scaled {
                        let a = allow[tok];
                        let excess = (c - a).max(0.0);
                        *score -= cfg.frequency * excess;
                        if cfg.deficit > 0.0 {
                            let rem = (length - pos) as f64 / length as f64;
                            *score += cfg.deficit * rem * (a - c).max(0.0);
                        }
                    } else {
                        *score -= cfg.frequency * c;
                    }
                }
            }
            if cfg.window > 0 {
                let start = history.len().saturating_sub(cfg.window);
                for (dist, &prev_tok) in history[start..].iter().rev().enumerate() {
                    step[prev_tok] -= cfg.strength * cfg.decay.powi(dist as i32);
                }
            }
            let mut best_idx = 0usize;
            let mut best_val = f64::NEG_INFINITY;
            for (idx, &val) in step.iter().enumerate() {
                if val > best_val || (val.is_nan() && !best_val.is_nan()) {
                    best_val = val;
                    best_idx = idx;
                }
            }
            out.push(best_idx as u32);
            history.push(best_idx);
            counts[best_idx] += 1.0;
        }
    }

    Ok(Tensor::from_vec(out, (batch, length), logits.device())?)
}

#[derive(Debug, Clone)]
pub enum DatasetItem {
    Text(String),
    Ids(Vec<u32>),
}

#[derive(Debug, Clone)]
pub struct InputExample {
    pub index: usize,
    pub prompt: DatasetItem,
    pub reference: DatasetItem,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrajectoryStep {
    pub level: usize,
    pub codes: usize,
    pub text: String,
    pub raw_ids: Vec<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SampleRecord {
    pub task: String,
    pub seed: u64,
    pub batch_size: usize,
    pub precision: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_ids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_truncated: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_ids: Option<Vec<u32>>,
    pub output: String,
    pub raw_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trajectory: Option<Vec<TrajectoryStep>>,
}

pub struct Generator {
    pub config: ModelEvalConfig,
    pub pre_tokenizer: Box<dyn PreTokenizer>,
    pub vqvae: Arc<Vqvae>,
    pub model: NextConceptTransformer,
    pub picker: TokenPickerConfig,
    pub initial_probabilities: Option<Vec<f32>>,
    pub eos: Option<u32>,
    pub pad: Option<u32>,
    pub cut_at_eos: bool,
    pub device: Device,
    pub dtype: DType,
    pub precision_label: String,
}

impl Generator {
    pub fn load(
        config: ModelEvalConfig,
        checkpoint_dir: &Path,
        device: Device,
        dtype: DType,
        precision_label: &str,
    ) -> Result<Self> {
        let pre_tokenizer =
            build_pre_tokenizer(&config.tokenizer.pre_tokenizer, Some(checkpoint_dir))?;

        let tok_path = checkpoint_dir.join(&config.checkpoints.tokenizer);
        let tok_pth = PthTensors::new(&tok_path, Some("model")).with_context(|| {
            format!("Failed to load tokenizer checkpoint {}", tok_path.display())
        })?;
        let tok_vb = VarBuilder::from_backend(Box::new(tok_pth), dtype, device.clone());
        let vqvae = Arc::new(Vqvae::load(&config.tokenizer.vqvae, tok_vb)?);

        let gen_path = checkpoint_dir.join(&config.checkpoints.generator);
        let gen_pth = PthTensors::new(&gen_path, Some("model")).with_context(|| {
            format!("Failed to load generator checkpoint {}", gen_path.display())
        })?;
        let gen_vb = VarBuilder::from_backend(Box::new(gen_pth), dtype, device.clone());
        let model = NextConceptTransformer::load(Arc::clone(&vqvae), config.model.clone(), gen_vb)?;

        let allowance = if let Some(ref spec) = config.generation.frequency_allowance {
            let t = load_tensor_spec(spec, checkpoint_dir, &config.checkpoints.generator)?;
            Some(t.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?)
        } else {
            None
        };

        let initial_probabilities = if let Some(ref spec) = config.generation.level0_unigram {
            let t = load_tensor_spec(spec, checkpoint_dir, &config.checkpoints.generator)?;
            Some(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
        } else {
            None
        };

        let picker = TokenPickerConfig::from_settings(
            config.generation.frequency_penalty,
            config.generation.frequency_deficit,
            config.generation.adjacency_penalty.as_deref(),
            allowance,
        );

        let eos = pre_tokenizer.eos_token_id();
        let pad = pre_tokenizer.pad_token_id();
        let cut_at_eos = config
            .generation
            .cut_at_eos
            .unwrap_or_else(|| config.eval.metric.as_deref() != Some("genppl_flm"));

        Ok(Self {
            config,
            pre_tokenizer,
            vqvae,
            model,
            picker,
            initial_probabilities,
            eos,
            pad,
            cut_at_eos,
            device,
            dtype,
            precision_label: precision_label.to_string(),
        })
    }

    pub fn decode_ids(&self, ids: &[u32], cut_at_eos: Option<bool>) -> Result<String> {
        let cut = cut_at_eos.unwrap_or(self.cut_at_eos);
        let mut slice = ids;
        if cut
            && let Some(eos) = self.eos
            && let Some(pos) = slice.iter().position(|&id| id == eos)
        {
            slice = &slice[..pos];
        }
        let filtered: Vec<u32> = slice
            .iter()
            .copied()
            .filter(|&id| Some(id) != self.eos && Some(id) != self.pad)
            .collect();
        self.pre_tokenizer.decode(&filtered, true)
    }

    pub fn generate_samples(
        &self,
        gen_cfg: &GenerationFileConfig,
        inputs: Option<&[InputExample]>,
        trajectory: bool,
        use_kv_cache: bool,
        rng: &mut ChaCha8Rng,
    ) -> Result<Vec<SampleRecord>> {
        let prefix_len = self.config.model.prefix_len;
        let l0_len = self.vqvae.vq.levels[0];
        let mut all_records = Vec::with_capacity(gen_cfg.samples);

        let opts = GenerateOptions {
            temperature: self.config.generation.temperature,
            top_k: self.config.generation.top_k,
            top_p: self.config.generation.top_p,
            use_kv_cache,
            cfg_scale: self.config.generation.cfg_scale,
            cfg_null_eos: self.eos.unwrap_or(50_256),
            soft_decisions: self.config.generation.soft_decisions,
            soft_temperature: self.config.generation.soft_temperature,
            return_trajectory: trajectory,
        };

        for start in (0..gen_cfg.samples).step_by(gen_cfg.batch_size) {
            let size = (gen_cfg.samples - start).min(gen_cfg.batch_size);
            let mut prefixes_flat = Vec::with_capacity(size * prefix_len);
            let mut batch_meta = Vec::with_capacity(size);

            for offset in 0..size {
                if let Some(items) = inputs {
                    let ex = &items[start + offset];
                    let ids = match &ex.prompt {
                        DatasetItem::Text(s) => self.pre_tokenizer.encode(s)?,
                        DatasetItem::Ids(v) => v.clone(),
                    };
                    let truncated = ids.len() > prefix_len;
                    let prompt_ids: Vec<u32> = ids.into_iter().take(prefix_len).collect();
                    let prompt_text = self.decode_ids(&prompt_ids, Some(false))?;
                    let mut padded = prompt_ids.clone();
                    let pad_id = self.pad.unwrap_or(0);
                    padded.resize(prefix_len, pad_id);
                    prefixes_flat.extend_from_slice(&padded);

                    let (ref_text, ref_ids) = match &ex.reference {
                        DatasetItem::Text(s) => {
                            let r_ids = self.pre_tokenizer.encode(s)?;
                            (s.clone(), r_ids)
                        }
                        DatasetItem::Ids(v) => (self.decode_ids(v, None)?, v.clone()),
                    };

                    batch_meta.push(Some((
                        ex.index,
                        prompt_ids,
                        prompt_text,
                        truncated,
                        ref_text,
                        ref_ids,
                    )));
                } else {
                    let eos = self.eos.unwrap_or(0);
                    prefixes_flat.push(eos);
                    batch_meta.push(None);
                }
            }

            let p_cols = if inputs.is_some() { prefix_len } else { 1 };
            let prefix_t = Tensor::from_vec(prefixes_flat, (size, p_cols), &self.device)?;

            let initial = if inputs.is_none()
                && let Some(ref probs) = self.initial_probabilities
            {
                let sampled = sample_unigram(probs, size * l0_len, rng);
                Some(Tensor::from_vec(sampled, (size, l0_len), &self.device)?)
            } else {
                None
            };

            let (_, latent) = self
                .model
                .generate(initial.as_ref(), &prefix_t, &opts, rng)?;

            let final_latent = if trajectory {
                let k = self.vqvae.vq.levels.len();
                latent.narrow(1, k - 1, 1)?.squeeze(1)?
            } else {
                latent.clone()
            };
            let logits = self
                .vqvae
                .decoder
                .forward(&final_latent.to_dtype(self.dtype)?)?;
            let tokens = pick_tokens(&logits, &self.picker)?;
            let token_rows = tokens.to_vec2::<u32>()?;

            let mut traj_rows: Vec<Vec<TrajectoryStep>> = vec![Vec::new(); size];
            if trajectory {
                let n_levels = self.vqvae.vq.levels.len();
                for (lvl_idx, &codes) in self.vqvae.vq.levels.iter().enumerate() {
                    let rows_at_lvl = if lvl_idx == n_levels - 1 {
                        token_rows.clone()
                    } else {
                        let lat_l = latent.narrow(1, lvl_idx, 1)?.squeeze(1)?;
                        let dec_l = self.vqvae.decoder.forward(&lat_l.to_dtype(self.dtype)?)?;
                        pick_tokens(&dec_l, &self.picker)?.to_vec2::<u32>()?
                    };
                    for (bi, row) in rows_at_lvl.into_iter().enumerate() {
                        let text = self.decode_ids(&row, None)?;
                        traj_rows[bi].push(TrajectoryStep {
                            level: lvl_idx + 1,
                            codes,
                            text,
                            raw_ids: row,
                        });
                    }
                }
            }

            for (bi, row) in token_rows.into_iter().enumerate() {
                let output = self.decode_ids(&row, None)?;
                let meta = batch_meta[bi].take();
                let (dataset_index, prompt_ids, prompt, prompt_truncated, reference, reference_ids) =
                    match meta {
                        Some((idx, pids, ptxt, trunc, rtxt, rids)) => (
                            Some(idx),
                            Some(pids),
                            Some(ptxt),
                            Some(trunc),
                            Some(rtxt),
                            Some(rids),
                        ),
                        None => (None, None, None, None, None, None),
                    };
                all_records.push(SampleRecord {
                    task: self.config.task.clone(),
                    seed: gen_cfg.seed,
                    batch_size: gen_cfg.batch_size,
                    precision: self.precision_label.clone(),
                    dataset_index,
                    prompt_ids,
                    prompt,
                    prompt_truncated,
                    reference,
                    reference_ids,
                    output,
                    raw_ids: row,
                    trajectory: if trajectory {
                        Some(std::mem::take(&mut traj_rows[bi]))
                    } else {
                        None
                    },
                });
            }
        }

        Ok(all_records)
    }
}

fn sample_unigram(probs: &[f32], n: usize, rng: &mut ChaCha8Rng) -> Vec<u32> {
    let total: f64 = probs.iter().map(|&p| f64::from(p)).sum();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut draw = rng.random::<f64>() * total;
        let mut chosen = 0u32;
        for (i, &p) in probs.iter().enumerate() {
            draw -= f64::from(p);
            if draw <= 0.0 {
                chosen = i as u32;
                break;
            }
        }
        out.push(chosen);
    }
    out
}

fn load_tensor_spec(spec: &str, checkpoint_dir: &Path, default_file: &str) -> Result<Tensor> {
    let (file_part, key_part) = match spec.split_once(':') {
        Some((f, k)) => (f, Some(k)),
        None => (spec, None),
    };
    let path = checkpoint_dir.join(if file_part.is_empty() {
        default_file
    } else {
        file_part
    });
    let pth = PthTensors::new(&path, key_part)
        .with_context(|| format!("Failed to load tensor spec {spec} from {}", path.display()))?;
    let names = pth.tensor_infos().keys().cloned().collect::<Vec<_>>();
    let first = names
        .first()
        .ok_or_else(|| anyhow::anyhow!("No tensor found in {}", path.display()))?;
    pth.get(first)?
        .ok_or_else(|| anyhow::anyhow!("Missing tensor {first}"))
}

pub fn load_inputs(
    data_dir: &Path,
    model_cfg: &ModelEvalConfig,
    split: &str,
    samples: usize,
    rng: &mut ChaCha8Rng,
) -> Result<Option<Vec<InputExample>>> {
    let data_file = match split {
        "test" => model_cfg.data.test.as_deref(),
        "validation" => model_cfg.data.validation.as_deref(),
        "train" => model_cfg.data.train.as_deref(),
        _ => None,
    };
    let Some(items_rel) = data_file else {
        return Ok(None);
    };
    let refs_rel = model_cfg
        .data
        .test_refs
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("Missing test_refs in data config"))?;

    let items = load_pickle_dataset(&data_dir.join(items_rel))?;
    let refs = load_pickle_dataset(&data_dir.join(refs_rel))?;
    if items.len() != refs.len() {
        bail!(
            "Input and reference counts differ ({} vs {})",
            items.len(),
            refs.len()
        );
    }
    if samples > items.len() {
        bail!("Requested {samples} samples from {} inputs", items.len());
    }
    let chosen = rand_sample_indices(rng, items.len(), samples);
    Ok(Some(
        chosen
            .into_iter()
            .map(|idx| InputExample {
                index: idx,
                prompt: items[idx].clone(),
                reference: refs[idx].clone(),
            })
            .collect(),
    ))
}

fn load_pickle_dataset(path: &Path) -> Result<Vec<DatasetItem>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open dataset pickle {}", path.display()))?;
    let val: Value = serde_pickle::from_reader(BufReader::new(file), DeOptions::default())
        .with_context(|| format!("Failed to parse dataset pickle {}", path.display()))?;
    let Value::List(list) = val else {
        bail!("Dataset pickle {} is not a list", path.display());
    };
    let mut out = Vec::with_capacity(list.len());
    for item in list {
        match item {
            Value::String(s) => out.push(DatasetItem::Text(s)),
            Value::List(ids) => {
                let mut vec = Vec::with_capacity(ids.len());
                for id in ids {
                    match id {
                        Value::I64(n) => vec.push(n as u32),
                        other => bail!("Unexpected token id in dataset pickle: {other:?}"),
                    }
                }
                out.push(DatasetItem::Ids(vec));
            }
            other => bail!("Unexpected dataset entry type: {other:?}"),
        }
    }
    Ok(out)
}
