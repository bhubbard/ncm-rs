use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CheckpointsConfig {
    pub generator: String,
    pub tokenizer: String,
    pub bpe_vocab: Option<String>,
    pub bpe_merges: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct DataConfig {
    pub test: Option<String>,
    pub test_refs: Option<String>,
    pub validation: Option<String>,
    pub train: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CtxEncoderConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_ctx_layers")]
    pub n_layer: usize,
    pub vocab_size: Option<usize>,
    pub pretrained: Option<String>,
    #[serde(default)]
    pub finetune: bool,
    #[serde(default)]
    pub pad_id: i64,
}

fn default_ctx_layers() -> usize {
    4
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct PostUpsampleConvConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_kernel_size")]
    pub kernel_size: usize,
}

fn default_kernel_size() -> usize {
    3
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TransformerConfig {
    pub n_layer: usize,
    pub n_head: usize,
    #[serde(default)]
    pub n_kv_heads: Option<usize>,
    #[serde(default = "default_true")]
    pub bias: bool,
    #[serde(default)]
    pub dropout: f64,
    pub n_embd: usize,
    pub context_length: usize,
    pub prefix_len: usize,
    #[serde(default)]
    pub raw_token_prefix: bool,
    #[serde(default)]
    pub contextual_prefix: bool,
    pub prefix_pad_token_id: Option<i64>,
    pub vocab_size: Vec<usize>,
    #[serde(default)]
    pub predict_level0: bool,
    #[serde(default = "default_attn_pattern")]
    pub attn_pattern: String,
    #[serde(default = "default_true")]
    pub use_positional_encoding: bool,
    #[serde(default = "default_true")]
    pub use_level_encoding: bool,
    #[serde(default)]
    pub use_rope: bool,
    #[serde(default = "default_rope_base")]
    pub rope_base: usize,
    #[serde(default)]
    pub use_qk_norm: bool,
    #[serde(default)]
    pub use_swiglu: bool,
    #[serde(default)]
    pub use_geglu: bool,
    #[serde(default)]
    pub use_relu2: bool,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,
    #[serde(default)]
    pub parallel_block: bool,
    #[serde(default)]
    pub normformer: bool,
    #[serde(default)]
    pub use_layerscale: bool,
    #[serde(default = "default_layerscale_init")]
    pub layerscale_init: f64,
    #[serde(default)]
    pub use_rezero: bool,
    #[serde(default)]
    pub ctx_encoder: Option<CtxEncoderConfig>,
    #[serde(default)]
    pub post_upsample_conv: Option<PostUpsampleConvConfig>,
    #[serde(default)]
    pub use_flex_attention: bool,
    #[serde(default)]
    pub query_level0: bool,
    #[serde(default)]
    pub l0_autoregressive: bool,
    #[serde(default)]
    pub shared_output_head: bool,
    #[serde(default)]
    pub shared_head_per_level_bias: bool,
    #[serde(default = "default_true")]
    pub untied_head_bias: bool,
    #[serde(default)]
    pub tie_heads_to_codebook: bool,
    #[serde(default = "default_true")]
    pub tie_heads_learn_scale: bool,
    #[serde(default)]
    pub tie_heads_trainable: bool,
    #[serde(default)]
    pub tie_heads_bias: bool,
}

fn default_true() -> bool {
    true
}
fn default_attn_pattern() -> String {
    "block_diagonal".to_string()
}
fn default_rope_base() -> usize {
    10_000
}
fn default_mlp_ratio() -> f64 {
    4.0
}
fn default_layerscale_init() -> f64 {
    1e-4
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct PreTokenizerConfig {
    pub name: String,
    #[serde(default)]
    pub tokenizer: Option<String>,
    pub model_id: Option<String>,
    pub vocab_file_path: Option<String>,
    pub merges_file_path: Option<String>,
    #[serde(default)]
    pub special_tokens: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LevelsConfig {
    #[serde(default)]
    pub use_manual_levels: bool,
    #[serde(default)]
    pub manual_levels: Vec<usize>,
    #[serde(default = "default_scheduler")]
    pub scheduler: String,
    #[serde(default = "default_one")]
    pub start: usize,
    #[serde(default = "default_64")]
    pub end: usize,
    #[serde(default = "default_8")]
    pub num_levels: usize,
}

fn default_scheduler() -> String {
    "linear".to_string()
}
fn default_one() -> usize {
    1
}
fn default_64() -> usize {
    64
}
fn default_8() -> usize {
    8
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum RatioSpec {
    Scalar(f64),
    PerLevel(Vec<f64>),
}

impl Default for RatioSpec {
    fn default() -> Self {
        Self::Scalar(0.5)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QuantResiConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub ratio: RatioSpec,
    #[serde(default)]
    pub share_mode: usize,
    #[serde(default)]
    pub learnable_ratio: bool,
    #[serde(default = "default_arch")]
    pub arch: String,
    #[serde(default = "default_one")]
    pub hidden_mult: usize,
    #[serde(default = "default_kernel_size")]
    pub kernel_size: usize,
}

fn default_arch() -> String {
    "blend".to_string()
}

impl Default for QuantResiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ratio: RatioSpec::Scalar(0.5),
            share_mode: 0,
            learnable_ratio: false,
            arch: "blend".to_string(),
            hidden_mult: 1,
            kernel_size: 3,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VectorQuantizerConfig {
    #[serde(default = "default_vq_class")]
    pub clss: String,
    #[serde(default = "default_decay")]
    pub decay: f64,
    #[serde(default = "default_eps")]
    pub epsilon: f64,
    #[serde(default = "default_commitment")]
    pub commitment_cost: f64,
    #[serde(default)]
    pub learned_l1_sampling: bool,
    #[serde(default)]
    pub learned_all_sampling: bool,
    #[serde(default)]
    pub learned_l1_cascade: bool,
    #[serde(default = "default_cascade_base")]
    pub learned_l1_cascade_base: usize,
    #[serde(default)]
    pub learned_l1_cascade_norm: bool,
    #[serde(default)]
    pub learned_l1_cascade_norm_final: bool,
    #[serde(default)]
    pub quant_resi: Option<QuantResiConfig>,
    pub levels: LevelsConfig,
}

fn default_vq_class() -> String {
    "multiscale_residual_vector_quantizer".to_string()
}
fn default_decay() -> f64 {
    0.99
}
fn default_eps() -> f64 {
    1e-5
}
fn default_commitment() -> f64 {
    0.25
}
fn default_cascade_base() -> usize {
    4
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VQVAEConfig {
    pub n_layers: usize,
    pub embed_dim: usize,
    pub in_vocab_size: usize,
    pub context_length: usize,
    #[serde(default = "default_one")]
    pub compression_factor: usize,
    #[serde(default)]
    pub dropout: f64,
    pub pre_quant_groupnorm: Option<usize>,
    #[serde(default = "default_norm_type")]
    pub pre_quant_norm_type: String,
    pub pre_quant_dropout: Option<f64>,
    #[serde(default)]
    pub use_qk_norm: bool,
    #[serde(default)]
    pub enc_layers: usize,
    #[serde(default)]
    pub enc_attn_window: usize,
    #[serde(default)]
    pub enc_conv_layers: usize,
    #[serde(default = "default_kernel_size")]
    pub enc_conv_kernel: usize,
    #[serde(default = "default_one")]
    pub dec_layers: usize,
    #[serde(default = "default_true")]
    pub enc_positional: bool,
    #[serde(default)]
    pub dec_positional: bool,
    pub vqvae_vocab_size: Vec<usize>,
    pub checkpoint_path: Option<String>,
    pub vector_quantizer_config: VectorQuantizerConfig,
}

fn default_norm_type() -> String {
    "group".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TokenizerConfig {
    pub pre_tokenizer: PreTokenizerConfig,
    pub vqvae: VQVAEConfig,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct GenerationSettings {
    #[serde(default = "default_temp")]
    pub temperature: f64,
    #[serde(default = "default_one")]
    pub top_k: usize,
    #[serde(default)]
    pub top_p: f64,
    #[serde(default)]
    pub cfg_scale: f64,
    #[serde(default)]
    pub soft_decisions: bool,
    #[serde(default = "default_temp")]
    pub soft_temperature: f64,
    #[serde(default)]
    pub cut_at_eos: Option<bool>,
    #[serde(default)]
    pub frequency_penalty: f64,
    #[serde(default)]
    pub frequency_deficit: f64,
    pub frequency_allowance: Option<String>,
    pub adjacency_penalty: Option<String>,
    pub level0_unigram: Option<String>,
}

fn default_temp() -> f64 {
    1.0
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct EvalSettings {
    pub metric: Option<String>,
    pub batch_size: Option<usize>,
    pub split: Option<String>,
    pub n_docs: Option<usize>,
    pub n_samples: Option<usize>,
    #[serde(default)]
    pub seeds: Vec<u64>,
    pub ref_model: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelEvalConfig {
    pub task: String,
    pub checkpoints: CheckpointsConfig,
    #[serde(default)]
    pub data: DataConfig,
    pub model: TransformerConfig,
    pub tokenizer: TokenizerConfig,
    #[serde(default)]
    pub generation: GenerationSettings,
    #[serde(default)]
    pub eval: EvalSettings,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenerationFileConfig {
    pub model_config: String,
    pub checkpoint_dir: Option<String>,
    pub data_dir: Option<String>,
    #[serde(default = "default_split")]
    pub split: String,
    #[serde(default = "default_one")]
    pub samples: usize,
    #[serde(default = "default_one")]
    pub batch_size: usize,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub generation: Option<serde_yaml::Value>,
}

fn default_split() -> String {
    "test".to_string()
}

/// Round continuous level lengths to strictly increasing integers between `start` and `end`.
#[must_use]
pub fn round_levels(levels: &[f64], start: usize, end: usize) -> Vec<usize> {
    let mut rounded = Vec::with_capacity(levels.len());
    rounded.push(start);
    for (i, &val) in levels
        .iter()
        .enumerate()
        .take(levels.len().saturating_sub(1))
        .skip(1)
    {
        let prev = rounded[i - 1];
        let fl = val.floor() as usize;
        let ce = val.ceil() as usize;
        if fl > prev && ce < end {
            rounded.push(fl);
        } else if fl <= prev && ce > prev && ce < end {
            rounded.push(ce);
        } else {
            rounded.push(prev + 1);
        }
    }
    rounded.push(end);
    rounded
}

/// Compute discrete level lengths from a named scheduler.
pub fn compute_levels(scheduler: &str, start: usize, end: usize, n: usize) -> Result<Vec<usize>> {
    if n < 2 {
        bail!("num_levels must be >= 2, got {n}");
    }
    let nf = (n - 1) as f64;
    let span = (end - start) as f64;
    let start_f = start as f64;
    let raw: Vec<f64> = match scheduler {
        "linear" => (0..n).map(|i| start_f + span * (i as f64) / nf).collect(),
        "cubic" => (0..n)
            .map(|i| start_f + span * (i as f64).powi(3) / nf.powi(3))
            .collect(),
        "exp" => (0..n)
            .map(|i| start_f + span * (i as f64).exp() / nf.exp())
            .collect(),
        "sqrt" => (0..n)
            .map(|i| start_f + span * (i as f64).sqrt() / nf.sqrt())
            .collect(),
        "sine" => (0..n)
            .map(|i| start_f + span * (std::f64::consts::FRAC_PI_2 * (i as f64 / nf)).sin())
            .collect(),
        "log" => {
            let n_ln = (n as f64).ln();
            (1..=n)
                .map(|i| start_f + span * (i as f64).ln() / n_ln)
                .collect()
        }
        "quadratic" => (0..n)
            .map(|i| start_f + span * (i as f64).powi(2) / nf.powi(2))
            .collect(),
        other => bail!("Unknown level scheduler: {other}"),
    };
    Ok(round_levels(&raw, start, end))
}

impl LevelsConfig {
    pub fn resolve(&self) -> Result<Vec<usize>> {
        if self.use_manual_levels {
            Ok(self.manual_levels.clone())
        } else {
            compute_levels(&self.scheduler, self.start, self.end, self.num_levels)
        }
    }
}

pub fn load_model_config(path: &Path) -> Result<ModelEvalConfig> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read model config {}", path.display()))?;
    serde_yaml::from_str(&text)
        .with_context(|| format!("Failed to parse model config {}", path.display()))
}

pub fn load_generation_config(
    path: &Path,
) -> Result<(GenerationFileConfig, ModelEvalConfig, PathBuf)> {
    let abs_path = std::fs::canonicalize(path)
        .with_context(|| format!("Failed to resolve generation config {}", path.display()))?;
    let text = std::fs::read_to_string(&abs_path)?;
    let gen_cfg: GenerationFileConfig = serde_yaml::from_str(&text)
        .with_context(|| format!("Failed to parse generation config {}", abs_path.display()))?;
    let parent = abs_path
        .parent()
        .context("Config has no parent directory")?;
    let model_cfg_path = parent.join(&gen_cfg.model_config);
    let mut model_cfg = load_model_config(&model_cfg_path)?;

    if let Some(ref overrides) = gen_cfg.generation {
        let mut base = serde_yaml::to_value(&model_cfg.generation)?;
        merge_yaml(&mut base, overrides);
        model_cfg.generation = serde_yaml::from_value(base)?;
    }

    Ok((gen_cfg, model_cfg, abs_path))
}

fn merge_yaml(base: &mut serde_yaml::Value, overlay: &serde_yaml::Value) {
    match (base, overlay) {
        (serde_yaml::Value::Mapping(b), serde_yaml::Value::Mapping(o)) => {
            for (k, v) in o {
                if let Some(existing) = b.get_mut(k) {
                    merge_yaml(existing, v);
                } else {
                    b.insert(k.clone(), v.clone());
                }
            }
        }
        (b, o) => {
            *b = o.clone();
        }
    }
}
