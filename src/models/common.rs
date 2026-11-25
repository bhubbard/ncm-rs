use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear, linear_no_bias, ops};

use crate::config::TransformerConfig;

pub fn linear_with_bias(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}

pub fn precompute_rope_cos_sin(
    positions: &[u32],
    head_dim: usize,
    base: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    if !head_dim.is_multiple_of(2) {
        candle_core::bail!("head_dim must be even for RoPE, got {head_dim}");
    }
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / (base as f32).powf(i as f32 / half as f32))
        .collect();
    let pos_f: Vec<f32> = positions.iter().map(|&p| p as f32).collect();
    let t_len = positions.len();
    let pos_t = Tensor::from_vec(pos_f, (t_len, 1), device)?;
    let inv_t = Tensor::from_vec(inv_freq, (1, half), device)?;
    let freqs = pos_t.matmul(&inv_t)?;
    let cos_half = freqs.cos()?;
    let sin_half = freqs.sin()?;
    let cos = Tensor::cat(&[&cos_half, &cos_half], D::Minus1)?;
    let sin = Tensor::cat(&[&sin_half, &sin_half], D::Minus1)?;
    Ok((cos, sin))
}

pub fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let cos = cos.to_dtype(dtype)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.to_dtype(dtype)?.unsqueeze(0)?.unsqueeze(0)?;
    let hd = x.dim(D::Minus1)?;
    let half = hd / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?
}

#[derive(Debug, Clone)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    eps: f64,
}

impl LayerNorm {
    pub fn load(ndim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(ndim, "weight")?;
        let bias = if bias {
            Some(vb.get(ndim, "bias")?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            eps: 1e-6,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f = x.to_dtype(DType::F32)?;
        let mean = x_f.mean_keepdim(D::Minus1)?;
        let diff = x_f.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(D::Minus1)?;
        let inv_std = (var + self.eps)?.sqrt()?.recip()?;
        let norm = diff.broadcast_mul(&inv_std)?.to_dtype(orig_dtype)?;
        let scaled = norm.broadcast_mul(&self.weight)?;
        match &self.bias {
            Some(b) => scaled.broadcast_add(b),
            None => Ok(scaled),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps: 1e-6 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f = x.to_dtype(DType::F32)?;
        let mean_sq = x_f.sqr()?.mean_keepdim(D::Minus1)?;
        let rsqrt = (mean_sq + self.eps)?.sqrt()?.recip()?;
        let normed = x_f.broadcast_mul(&rsqrt)?.to_dtype(orig_dtype)?;
        normed.broadcast_mul(&self.weight)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMode {
    Causal,
    Bidirectional,
    Window(usize),
}

#[derive(Debug, Clone)]
pub struct SelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_kv_head: usize,
    n_embd: usize,
    kv_dim: usize,
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,
    mode: AttentionMode,
}

impl SelfAttention {
    pub fn load(cfg: &TransformerConfig, mode: AttentionMode, vb: VarBuilder) -> Result<Self> {
        let n_head = cfg.n_head;
        let n_kv_head = cfg.n_kv_heads.unwrap_or(n_head);
        let n_embd = cfg.n_embd;
        let hd = n_embd / n_head;
        let kv_dim = n_kv_head * hd;
        let c_attn = linear_with_bias(n_embd, n_embd + 2 * kv_dim, cfg.bias, vb.pp("c_attn"))?;
        let c_proj = linear_with_bias(n_embd, n_embd, cfg.bias, vb.pp("c_proj"))?;
        let (q_norm, k_norm) = if cfg.use_qk_norm {
            (
                Some(RMSNorm::load(hd, vb.pp("q_norm"))?),
                Some(RMSNorm::load(hd, vb.pp("k_norm"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            c_attn,
            c_proj,
            n_head,
            n_kv_head,
            n_embd,
            kv_dim,
            q_norm,
            k_norm,
            mode,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        rope_cos_sin: Option<&(Tensor, Tensor)>,
        past_kv: Option<&(Tensor, Tensor)>,
        return_kv: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let (b, t, c) = x.dims3()?;
        let hd = c / self.n_head;
        let qkv = self.c_attn.forward(x)?;
        let q = qkv.narrow(D::Minus1, 0, self.n_embd)?;
        let k = qkv.narrow(D::Minus1, self.n_embd, self.kv_dim)?;
        let v = qkv.narrow(D::Minus1, self.n_embd + self.kv_dim, self.kv_dim)?;

        let mut q = q
            .reshape((b, t, self.n_head, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut k = k
            .reshape((b, t, self.n_kv_head, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((b, t, self.n_kv_head, hd))?
            .transpose(1, 2)?
            .contiguous()?;

        if self.n_kv_head != self.n_head {
            let rep = self.n_head / self.n_kv_head;
            k = k.repeat((1, rep, 1, 1))?;
            v = v.repeat((1, rep, 1, 1))?;
        }

        if let (Some(qn), Some(kn)) = (&self.q_norm, &self.k_norm) {
            q = qn.forward(&q)?;
            k = kn.forward(&k)?;
        }

        if let Some((cos_full, sin_full)) = rope_cos_sin {
            let past_len = match past_kv {
                Some((pk, _)) => pk.dim(2)?,
                None => 0,
            };
            let cos = cos_full.narrow(0, past_len, t)?;
            let sin = sin_full.narrow(0, past_len, t)?;
            q = apply_rotary_emb(&q, &cos, &sin)?;
            k = apply_rotary_emb(&k, &cos, &sin)?;
        }

        let new_kv = if return_kv {
            Some((k.contiguous()?, v.contiguous()?))
        } else {
            None
        };

        if let Some((pk, pv)) = past_kv {
            k = Tensor::cat(&[pk, &k], 2)?;
            v = Tensor::cat(&[pv, &v], 2)?;
        }

        let scale = 1.0 / (hd as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;

        if let Some(mask) = attn_mask {
            scores = scores.broadcast_add(&mask.to_dtype(scores.dtype())?)?;
        } else {
            match self.mode {
                AttentionMode::Window(w) if w > 0 => {
                    let w_mask = build_window_mask(t, w, scores.device(), scores.dtype())?;
                    scores = scores.broadcast_add(&w_mask)?;
                }
                AttentionMode::Causal if t > 1 => {
                    let c_mask = build_causal_mask(t, scores.device(), scores.dtype())?;
                    scores = scores.broadcast_add(&c_mask)?;
                }
                AttentionMode::Causal | AttentionMode::Bidirectional | AttentionMode::Window(_) => {
                }
            }
        }

        let probs = ops::softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(v.dtype())?;
        let y = probs.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, c))?;
        let y = self.c_proj.forward(&y)?;
        Ok((y, new_kv))
    }
}

fn build_causal_mask(t: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let mut data = vec![0.0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            data[i * t + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (1, 1, t, t), device)?.to_dtype(dtype)
}

fn build_window_mask(t: usize, window: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let mut data = vec![0.0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            if i.abs_diff(j) > window {
                data[i * t + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, t, t), device)?.to_dtype(dtype)
}

#[derive(Debug, Clone)]
pub enum MlpKind {
    Gated {
        w_gate: Linear,
        w_up: Linear,
        c_proj: Linear,
        silu: bool,
    },
    Standard {
        c_fc: Linear,
        c_proj: Linear,
        relu2: bool,
    },
}

#[derive(Debug, Clone)]
pub struct Mlp {
    kind: MlpKind,
}

impl Mlp {
    pub fn load(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let gated = cfg.use_swiglu || cfg.use_geglu;
        let kind = if gated {
            let h = ((8.0 * cfg.n_embd as f64 / 3.0 / 8.0).round() * 8.0) as usize;
            let w_gate = linear_with_bias(cfg.n_embd, h, cfg.bias, vb.pp("w_gate"))?;
            let w_up = linear_with_bias(cfg.n_embd, h, cfg.bias, vb.pp("w_up"))?;
            let c_proj = linear_with_bias(h, cfg.n_embd, cfg.bias, vb.pp("c_proj"))?;
            MlpKind::Gated {
                w_gate,
                w_up,
                c_proj,
                silu: cfg.use_swiglu,
            }
        } else {
            let h = (cfg.mlp_ratio * cfg.n_embd as f64).round() as usize;
            let c_fc = linear_with_bias(cfg.n_embd, h, cfg.bias, vb.pp("c_fc"))?;
            let c_proj = linear_with_bias(h, cfg.n_embd, cfg.bias, vb.pp("c_proj"))?;
            MlpKind::Standard {
                c_fc,
                c_proj,
                relu2: cfg.use_relu2,
            }
        };
        Ok(Self { kind })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.kind {
            MlpKind::Gated {
                w_gate,
                w_up,
                c_proj,
                silu,
            } => {
                let g = w_gate.forward(x)?;
                let act = if *silu { ops::silu(&g)? } else { g.gelu_erf()? };
                let up = w_up.forward(x)?;
                c_proj.forward(&(act * up)?)
            }
            MlpKind::Standard {
                c_fc,
                c_proj,
                relu2,
            } => {
                let h = c_fc.forward(x)?;
                let act = if *relu2 {
                    h.relu()?.sqr()?
                } else {
                    h.gelu_erf()?
                };
                c_proj.forward(&act)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    ln_1: LayerNorm,
    attn: SelfAttention,
    ln_2: LayerNorm,
    mlp: Mlp,
    is_parallel: bool,
    post_norms: Option<(LayerNorm, LayerNorm)>,
    scale_1: Option<Tensor>,
    scale_2: Option<Tensor>,
}

impl Block {
    pub fn load(cfg: &TransformerConfig, mode: AttentionMode, vb: VarBuilder) -> Result<Self> {
        let ln_1 = LayerNorm::load(cfg.n_embd, cfg.bias, vb.pp("ln_1"))?;
        let attn = SelfAttention::load(cfg, mode, vb.pp("attn"))?;
        let ln_2 = LayerNorm::load(cfg.n_embd, cfg.bias, vb.pp("ln_2"))?;
        let mlp = Mlp::load(cfg, vb.pp("mlp"))?;

        let post_norms = if cfg.normformer {
            Some((
                LayerNorm::load(cfg.n_embd, cfg.bias, vb.pp("ln_post_attn"))?,
                LayerNorm::load(cfg.n_embd, cfg.bias, vb.pp("ln_post_mlp"))?,
            ))
        } else {
            None
        };

        let (scale_1, scale_2) = if cfg.use_layerscale {
            (
                Some(vb.get(cfg.n_embd, "ls1")?),
                Some(vb.get(cfg.n_embd, "ls2")?),
            )
        } else if cfg.use_rezero {
            (Some(vb.get(1, "rz1")?), Some(vb.get(1, "rz2")?))
        } else {
            (None, None)
        };

        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
            is_parallel: cfg.parallel_block,
            post_norms,
            scale_1,
            scale_2,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        rope_cos_sin: Option<&(Tensor, Tensor)>,
        past_kv: Option<&(Tensor, Tensor)>,
        return_kv: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        if self.is_parallel {
            let n = self.ln_1.forward(x)?;
            let (mut a, new_kv) =
                self.attn
                    .forward(&n, attn_mask, rope_cos_sin, past_kv, return_kv)?;
            let mut m = self.mlp.forward(&n)?;
            if let Some((pa, pm)) = &self.post_norms {
                a = pa.forward(&a)?;
                m = pm.forward(&m)?;
            }
            if let Some(s1) = &self.scale_1 {
                a = a.broadcast_mul(s1)?;
            }
            if let Some(s2) = &self.scale_2 {
                m = m.broadcast_mul(s2)?;
            }
            let out = ((x + a)? + m)?;
            return Ok((out, new_kv));
        }

        let (mut a, new_kv) = self.attn.forward(
            &self.ln_1.forward(x)?,
            attn_mask,
            rope_cos_sin,
            past_kv,
            return_kv,
        )?;
        if let Some((pa, _)) = &self.post_norms {
            a = pa.forward(&a)?;
        }
        if let Some(s1) = &self.scale_1 {
            a = a.broadcast_mul(s1)?;
        }
        let x = (x + a)?;
        let mut m = self.mlp.forward(&self.ln_2.forward(&x)?)?;
        if let Some((_, pm)) = &self.post_norms {
            m = pm.forward(&m)?;
        }
        if let Some(s2) = &self.scale_2 {
            m = m.broadcast_mul(s2)?;
        }
        Ok(((x + m)?, new_kv))
    }
}
