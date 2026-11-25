use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Embedding, Module, VarBuilder,
    conv_transpose1d, conv1d, embedding, ops,
};

use crate::config::{QuantResiConfig, RatioSpec, VectorQuantizerConfig};

pub fn group_norm_no_affine(x: &Tensor, num_groups: usize, eps: f64) -> Result<Tensor> {
    let (b, c, l) = x.dims3()?;
    let orig_dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let grouped = xf.reshape((b, num_groups, (c / num_groups) * l))?;
    let mean = grouped.mean_keepdim(D::Minus1)?;
    let diff = grouped.broadcast_sub(&mean)?;
    let var = diff.sqr()?.mean_keepdim(D::Minus1)?;
    let inv_std = (var + eps)?.sqrt()?.recip()?;
    let norm = diff.broadcast_mul(&inv_std)?;
    norm.reshape((b, c, l))?.to_dtype(orig_dtype)
}

pub fn build_resize_matrix(
    nin: usize,
    nout: usize,
    mode: ResizeMode,
    device: &Device,
) -> Result<Tensor> {
    let mut mat = vec![0.0f32; nout * nin];
    match mode {
        ResizeMode::Linear => {
            if nin == 1 {
                for row in 0..nout {
                    mat[row * nin] = 1.0;
                }
            } else {
                let scale = nin as f64 / nout as f64;
                for row in 0..nout {
                    let src = ((row as f64 + 0.5) * scale - 0.5).clamp(0.0, (nin - 1) as f64);
                    let i0 = src.floor() as usize;
                    let i1 = (i0 + 1).min(nin - 1);
                    let w1 = (src - i0 as f64) as f32;
                    let w0 = 1.0 - w1;
                    mat[row * nin + i0] += w0;
                    mat[row * nin + i1] += w1;
                }
            }
        }
        ResizeMode::Area => {
            for row in 0..nout {
                let start = (row * nin) / nout;
                let end = ((row + 1) * nin).div_ceil(nout);
                let count = (end - start).max(1) as f32;
                let w = 1.0 / count;
                for col in start..end {
                    mat[row * nin + col] = w;
                }
            }
        }
    }
    Tensor::from_vec(mat, (nout, nin), device)
}

#[derive(Debug, Clone, Copy)]
pub enum ResizeMode {
    Linear,
    Area,
}

#[derive(Debug, Clone)]
enum PhiUnit {
    Identity,
    Blend {
        conv: Conv1d,
        ratio_const: f64,
        ratio_logit: Option<Tensor>,
    },
    ConvStack {
        c1: Conv1d,
        c2: Conv1d,
        c3: Conv1d,
    },
}

impl PhiUnit {
    fn load(embed_dim: usize, ratio: f64, cfg: &QuantResiConfig, vb: VarBuilder) -> Result<Self> {
        if cfg.arch == "convstack" {
            let ks = cfg.kernel_size;
            let pad = ks / 2;
            let h = embed_dim * cfg.hidden_mult;
            let c_cfg = Conv1dConfig {
                padding: pad,
                ..Default::default()
            };
            return Ok(Self::ConvStack {
                c1: conv1d(embed_dim, h, ks, c_cfg, vb.pp("c1"))?,
                c2: conv1d(h, h, ks, c_cfg, vb.pp("c2"))?,
                c3: conv1d(h, embed_dim, ks, c_cfg, vb.pp("c3"))?,
            });
        }
        let init_ratio = ratio.abs();
        if !cfg.learnable_ratio && init_ratio <= 1e-6 {
            return Ok(Self::Identity);
        }
        let c_cfg = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv = conv1d(embed_dim, embed_dim, 3, c_cfg, vb.clone())?;
        let ratio_logit = if cfg.learnable_ratio {
            Some(vb.get((), "ratio_logit")?)
        } else {
            None
        };
        Ok(Self::Blend {
            conv,
            ratio_const: init_ratio,
            ratio_logit,
        })
    }

    fn forward(&self, h_bcl: &Tensor) -> Result<Tensor> {
        match self {
            Self::Identity => Ok(h_bcl.clone()),
            Self::Blend {
                conv,
                ratio_const,
                ratio_logit,
            } => {
                let c_out = conv.forward(h_bcl)?;
                if let Some(logit) = ratio_logit {
                    let r = ops::sigmoid(&logit.to_dtype(h_bcl.dtype())?)?;
                    let one_minus_r = (1.0 - &r)?;
                    h_bcl.broadcast_mul(&one_minus_r)? + c_out.broadcast_mul(&r)?
                } else {
                    (h_bcl * (1.0 - *ratio_const))? + (c_out * *ratio_const)?
                }
            }
            Self::ConvStack { c1, c2, c3 } => {
                let y = c1.forward(h_bcl)?.gelu_erf()?;
                let y = c2.forward(&y)?.gelu_erf()?;
                let y = c3.forward(&y)?;
                h_bcl + y
            }
        }
    }
}

#[derive(Debug, Clone)]
enum QuantResi {
    Shared(PhiUnit),
    NonShared {
        units: Vec<PhiUnit>,
        ticks: Vec<f64>,
    },
    PartiallyShared {
        units: Vec<PhiUnit>,
        ticks: Vec<f64>,
    },
}

fn make_ticks(k: usize) -> Vec<f64> {
    if k == 0 {
        return Vec::new();
    }
    if k == 1 {
        return vec![0.5];
    }
    let start = if k == 4 {
        1.0 / 3.0 / k as f64
    } else {
        1.0 / 2.0 / k as f64
    };
    let end = 1.0 - start;
    (0..k)
        .map(|i| start + (end - start) * (i as f64) / ((k - 1) as f64))
        .collect()
}

fn nearest_tick_idx(ticks: &[f64], at: f64) -> usize {
    let mut best_i = 0;
    let mut best_d = f64::INFINITY;
    for (i, &t) in ticks.iter().enumerate() {
        let d = (t - at).abs();
        if d < best_d {
            best_d = d;
            best_i = i;
        }
    }
    best_i
}

impl QuantResi {
    fn load(
        embed_dim: usize,
        num_levels: usize,
        cfg: &QuantResiConfig,
        vb: VarBuilder,
    ) -> Result<Option<Self>> {
        if !cfg.enabled {
            return Ok(None);
        }
        let ratios: Vec<f64> = match &cfg.ratio {
            RatioSpec::Scalar(r) => vec![*r; num_levels],
            RatioSpec::PerLevel(list) => list.clone(),
        };
        if cfg.share_mode == 0 {
            let mut units = Vec::with_capacity(num_levels);
            for (i, &r) in ratios.iter().enumerate() {
                units.push(PhiUnit::load(embed_dim, r, cfg, vb.pp(i))?);
            }
            return Ok(Some(Self::NonShared {
                ticks: make_ticks(num_levels),
                units,
            }));
        }
        if cfg.share_mode == 1 {
            let unit = PhiUnit::load(embed_dim, ratios[0], cfg, vb.pp("qresi"))?;
            return Ok(Some(Self::Shared(unit)));
        }
        let k = cfg.share_mode;
        let mut units = Vec::with_capacity(k);
        for i in 0..k {
            units.push(PhiUnit::load(
                embed_dim,
                ratios[0],
                cfg,
                vb.pp("qresi_ls").pp(i),
            )?);
        }
        Ok(Some(Self::PartiallyShared {
            ticks: make_ticks(k),
            units,
        }))
    }

    fn forward_at(&self, h_bcl: &Tensor, at: f64) -> Result<Tensor> {
        match self {
            Self::Shared(u) => u.forward(h_bcl),
            Self::NonShared { units, ticks } | Self::PartiallyShared { units, ticks } => {
                let idx = nearest_tick_idx(ticks, at);
                units[idx].forward(h_bcl)
            }
        }
    }
}

fn plan_cascade_strides(total_stride: usize, base: usize) -> Vec<usize> {
    let mut s = total_stride;
    let mut strides = Vec::new();
    while s > 1 {
        let mut f = base.min(s);
        while f > 1 && !s.is_multiple_of(f) {
            f -= 1;
        }
        if f <= 1 {
            strides.push(s);
            s = 1;
        } else {
            strides.push(f);
            s /= f;
        }
    }
    if strides.is_empty() { vec![1] } else { strides }
}

#[derive(Debug, Clone)]
enum CascadeOp {
    Down(Conv1d),
    Up(ConvTranspose1d),
    GroupNorm(usize),
}

impl CascadeOp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Down(c) => c.forward(x),
            Self::Up(c) => c.forward(x),
            Self::GroupNorm(g) => group_norm_no_affine(x, *g, 1e-5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultiscaleResidualVectorQuantizer {
    pub levels: Vec<usize>,
    pub embed_dim: usize,
    pub vqvae_vocab_sizes: Vec<usize>,
    pub embeddings: Vec<Embedding>,
    quant_resi: Option<QuantResi>,
    learned_l1_sampling: bool,
    learned_all_sampling: bool,
    learned_down_l1: Vec<CascadeOp>,
    learned_up_l1: Vec<CascadeOp>,
    learned_down_refine: Vec<Option<Conv1d>>,
    learned_up_refine: Vec<Option<Conv1d>>,
    up_matrices: Vec<Option<Tensor>>,
    down_matrices: Vec<Option<Tensor>>,
}

impl MultiscaleResidualVectorQuantizer {
    pub fn load(
        vqvae_vocab_sizes: &[usize],
        embed_dim: usize,
        cfg: &VectorQuantizerConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let levels = cfg
            .levels
            .resolve()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let num_levels = levels.len();

        let mut embeddings = Vec::with_capacity(num_levels);
        for (i, &vs) in vqvae_vocab_sizes.iter().enumerate() {
            embeddings.push(embedding(vs, embed_dim, vb.pp("embeddings").pp(i))?);
        }

        let qr_cfg = cfg.quant_resi.clone().unwrap_or_default();
        let quant_resi = QuantResi::load(embed_dim, num_levels, &qr_cfg, vb.pp("quant_resi"))?;

        let learned_all_sampling = cfg.learned_all_sampling;
        let learned_l1_sampling = cfg.learned_l1_sampling || learned_all_sampling;

        let mut learned_down_l1 = Vec::new();
        let mut learned_up_l1 = Vec::new();
        if learned_l1_sampling {
            let l_full = *levels.last().unwrap_or(&1);
            let ell_1 = levels[0];
            let stride = l_full / ell_1;
            if cfg.learned_l1_cascade {
                let down_strides = plan_cascade_strides(stride, cfg.learned_l1_cascade_base);
                let mut seq_idx = 0usize;
                for (i, &s) in down_strides.iter().enumerate() {
                    if cfg.learned_l1_cascade_norm && i > 0 {
                        learned_down_l1.push(CascadeOp::GroupNorm(8));
                        seq_idx += 1;
                    }
                    let c_cfg = Conv1dConfig {
                        stride: s,
                        ..Default::default()
                    };
                    learned_down_l1.push(CascadeOp::Down(conv1d(
                        embed_dim,
                        embed_dim,
                        s,
                        c_cfg,
                        vb.pp("learned_down_l1").pp(seq_idx),
                    )?));
                    seq_idx += 1;
                }
                if cfg.learned_l1_cascade_norm_final {
                    learned_down_l1.push(CascadeOp::GroupNorm(8));
                }

                let mut up_strides = down_strides;
                up_strides.reverse();
                let mut up_idx = 0usize;
                for (i, &s) in up_strides.iter().enumerate() {
                    if cfg.learned_l1_cascade_norm && i > 0 {
                        learned_up_l1.push(CascadeOp::GroupNorm(8));
                        up_idx += 1;
                    }
                    let ct_cfg = ConvTranspose1dConfig {
                        stride: s,
                        ..Default::default()
                    };
                    learned_up_l1.push(CascadeOp::Up(conv_transpose1d(
                        embed_dim,
                        embed_dim,
                        s,
                        ct_cfg,
                        vb.pp("learned_up_l1").pp(up_idx),
                    )?));
                    up_idx += 1;
                }
            } else {
                let c_cfg = Conv1dConfig {
                    stride,
                    ..Default::default()
                };
                learned_down_l1.push(CascadeOp::Down(conv1d(
                    embed_dim,
                    embed_dim,
                    stride,
                    c_cfg,
                    vb.pp("learned_down_l1"),
                )?));
                let ct_cfg = ConvTranspose1dConfig {
                    stride,
                    ..Default::default()
                };
                learned_up_l1.push(CascadeOp::Up(conv_transpose1d(
                    embed_dim,
                    embed_dim,
                    stride,
                    ct_cfg,
                    vb.pp("learned_up_l1"),
                )?));
            }
        }

        let mut learned_down_refine = Vec::new();
        let mut learned_up_refine = Vec::new();
        if learned_all_sampling {
            let c_cfg = Conv1dConfig {
                padding: 1,
                ..Default::default()
            };
            for i in 0..num_levels.saturating_sub(1) {
                if i >= 1 && i <= num_levels - 2 {
                    learned_down_refine.push(Some(conv1d(
                        embed_dim,
                        embed_dim,
                        3,
                        c_cfg,
                        vb.pp("learned_down_refine").pp(i),
                    )?));
                    learned_up_refine.push(Some(conv1d(
                        embed_dim,
                        embed_dim,
                        3,
                        c_cfg,
                        vb.pp("learned_up_refine").pp(i),
                    )?));
                } else {
                    learned_down_refine.push(None);
                    learned_up_refine.push(None);
                }
            }
        }

        let full_len = *levels.last().unwrap_or(&1);
        let dev = vb.device();
        let mut up_matrices = Vec::with_capacity(num_levels);
        let mut down_matrices = Vec::with_capacity(num_levels);
        for (i, &lvl) in levels.iter().enumerate() {
            if i < num_levels - 1 {
                up_matrices.push(Some(build_resize_matrix(
                    lvl,
                    full_len,
                    ResizeMode::Linear,
                    dev,
                )?));
            } else {
                up_matrices.push(None);
            }
            if i > 0 {
                down_matrices.push(Some(build_resize_matrix(
                    full_len,
                    lvl,
                    ResizeMode::Area,
                    dev,
                )?));
            } else {
                down_matrices.push(None);
            }
        }

        Ok(Self {
            levels,
            embed_dim,
            vqvae_vocab_sizes: vqvae_vocab_sizes.to_vec(),
            embeddings,
            quant_resi,
            learned_l1_sampling,
            learned_all_sampling,
            learned_down_l1,
            learned_up_l1,
            learned_down_refine,
            learned_up_refine,
            up_matrices,
            down_matrices,
        })
    }

    pub fn apply_quant_resi(&self, h_blc: &Tensor, level_idx: usize) -> Result<Tensor> {
        let Some(qr) = &self.quant_resi else {
            return Ok(h_blc.clone());
        };
        let h_bcl = h_blc.transpose(1, 2)?.contiguous()?;
        let denom = self.levels.len().saturating_sub(1).max(1) as f64;
        let at = (level_idx as f64) / denom;
        let out = qr.forward_at(&h_bcl, at)?;
        out.transpose(1, 2)?.contiguous()
    }

    pub fn down_for_quant(&self, z_blc: &Tensor, level_idx: usize) -> Result<Tensor> {
        let z_bcl = z_blc.transpose(1, 2)?;
        if level_idx == 0 && self.learned_l1_sampling {
            let mut h = z_bcl;
            for op in &self.learned_down_l1 {
                h = op.forward(&h)?;
            }
            return h.transpose(1, 2)?.contiguous();
        }
        let target_len = self.levels[level_idx];
        let full_len = *self.levels.last().unwrap_or(&1);
        let down_mat = if let Some(m) = &self.down_matrices[level_idx] {
            m.clone()
        } else {
            build_resize_matrix(full_len, target_len, ResizeMode::Area, z_blc.device())?
        };
        let mut zd_blc = down_mat.to_dtype(z_blc.dtype())?.broadcast_matmul(z_blc)?;
        if self.learned_all_sampling
            && let Some(Some(refine)) = self.learned_down_refine.get(level_idx)
        {
            let ch = zd_blc.transpose(1, 2)?;
            zd_blc = (ch.clone() + refine.forward(&ch)?)?.transpose(1, 2)?;
        }
        zd_blc.contiguous()
    }

    pub fn up_to_full(&self, q_blc: &Tensor, level_idx: usize) -> Result<Tensor> {
        if level_idx == self.levels.len() - 1 {
            return Ok(q_blc.clone());
        }
        if level_idx == 0 && self.learned_l1_sampling {
            let mut h = q_blc.transpose(1, 2)?;
            for op in &self.learned_up_l1 {
                h = op.forward(&h)?;
            }
            return h.transpose(1, 2)?.contiguous();
        }
        let up_mat = self.up_matrices[level_idx]
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("missing up_matrix".to_string()))?;
        let mut qu_blc = up_mat.to_dtype(q_blc.dtype())?.broadcast_matmul(q_blc)?;
        if self.learned_all_sampling
            && let Some(Some(refine)) = self.learned_up_refine.get(level_idx)
        {
            let ch = qu_blc.transpose(1, 2)?;
            qu_blc = (ch.clone() + refine.forward(&ch)?)?.transpose(1, 2)?;
        }
        qu_blc.contiguous()
    }

    pub fn z_to_idx(&self, z: &Tensor) -> Result<Tensor> {
        let z = z.to_dtype(DType::F32)?;
        let (b, _l, c) = z.dims3()?;
        let mut z_rest = z.clone();
        let mut all_indices = Vec::with_capacity(self.levels.len());
        let num_levels = self.levels.len();

        for (i, &level) in self.levels.iter().enumerate() {
            let flat_input_blc = if i == num_levels - 1 {
                z_rest.clone()
            } else {
                self.down_for_quant(&z_rest, i)?
            };
            let flat_input = flat_input_blc.reshape((b * level, c))?;
            let emb_w = self.embeddings[i].embeddings().to_dtype(DType::F32)?;

            let x_sq = flat_input.sqr()?.sum_keepdim(1)?;
            let e_sq = emb_w.sqr()?.sum_keepdim(1)?.transpose(0, 1)?;
            let dot = flat_input.matmul(&emb_w.transpose(0, 1)?)?;
            let dist = (x_sq.broadcast_add(&e_sq)? - (dot * 2.0)?)?;

            let indices = dist.argmin(D::Minus1)?.reshape((b, level))?;
            all_indices.push(indices.clone());

            let emb_q = self.embeddings[i].forward(&indices)?.to_dtype(DType::F32)?;
            let quantized = if i == num_levels - 1 {
                emb_q
            } else {
                self.up_to_full(&emb_q, i)?
            };
            let quantized = self.apply_quant_resi(&quantized, i)?;
            z_rest = (z_rest - quantized)?;
        }

        Tensor::cat(&all_indices, D::Minus1)
    }

    pub fn get_next_autoregressive_input(
        &self,
        level_idx: usize,
        z_hat: &Tensor,
        z_q_level: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let n_levels = self.levels.len();
        let upsampled = if level_idx == n_levels - 1 {
            z_q_level.clone()
        } else {
            self.up_to_full(z_q_level, level_idx)?
        };
        let upsampled = self.apply_quant_resi(&upsampled, level_idx)?;
        let z_hat_next = (z_hat + upsampled)?;
        if level_idx == n_levels - 1 {
            Ok((z_hat_next.clone(), z_hat_next))
        } else {
            let down_mat = self.down_matrices[level_idx + 1]
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("missing down_matrix".to_string()))?;
            let next_in = down_mat
                .to_dtype(z_hat_next.dtype())?
                .broadcast_matmul(&z_hat_next)?;
            Ok((z_hat_next, next_in))
        }
    }

    pub fn idx_to_z_q(&self, indices: &Tensor) -> Result<Tensor> {
        let b = indices.dim(0)?;
        let full_len = *self.levels.last().unwrap_or(&1);
        let mut z_hat = Tensor::zeros((b, full_len, self.embed_dim), DType::F32, indices.device())?;
        let mut offset = 0usize;
        for (i, &lvl) in self.levels.iter().enumerate() {
            let slice = indices.narrow(1, offset, lvl)?;
            let emb = self.embeddings[i].forward(&slice)?.to_dtype(DType::F32)?;
            let q = if i == self.levels.len() - 1 {
                emb
            } else {
                self.up_to_full(&emb, i)?
            };
            let q = self.apply_quant_resi(&q, i)?;
            z_hat = (z_hat + q)?;
            offset += lvl;
        }
        Ok(z_hat)
    }
}
