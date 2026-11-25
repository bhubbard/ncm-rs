use std::sync::Arc;

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, Embedding, Linear, Module, VarBuilder, conv1d, embedding, linear_no_bias,
    ops,
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::config::TransformerConfig;
use crate::models::common::{
    AttentionMode, Block, LayerNorm, linear_with_bias, precompute_rope_cos_sin,
};
use crate::models::vqvae::Vqvae;

#[derive(Debug, Clone)]
enum OutputHead {
    Linear(Linear),
    Shared {
        head: Linear,
        bias: Option<Tensor>,
    },
    TiedCodebook {
        proj: Linear,
        bias: Option<Tensor>,
        codebook: Tensor,
        e_sq: Tensor,
        d_scale: f64,
        log_inv_t: f64,
    },
}

impl OutputHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(l) => l.forward(x),
            Self::Shared { head, bias } => {
                let out = head.forward(x)?;
                match bias {
                    Some(b) => out.broadcast_add(b),
                    None => Ok(out),
                }
            }
            Self::TiedCodebook {
                proj,
                bias,
                codebook,
                e_sq,
                d_scale,
                log_inv_t,
            } => {
                let z = proj.forward(x)?;
                let e_t = codebook.to_dtype(z.dtype())?.transpose(0, 1)?;
                let dot = z.broadcast_matmul(&e_t)?;
                let neg_d2 = ((dot * 2.0)?.broadcast_sub(&e_sq.to_dtype(z.dtype())?)? / *d_scale)?;
                let mut out = (neg_d2 * log_inv_t.exp())?;
                if let Some(b) = bias {
                    out = out.broadcast_add(&b.to_dtype(out.dtype())?)?;
                }
                Ok(out)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub use_kv_cache: bool,
    pub cfg_scale: f64,
    pub cfg_null_eos: u32,
    pub soft_decisions: bool,
    pub soft_temperature: f64,
    pub return_trajectory: bool,
}

#[derive(Debug, Clone)]
pub struct NextConceptTransformer {
    pub config: TransformerConfig,
    pub vqvae: Arc<Vqvae>,
    pub levels: Vec<usize>,
    pub total_level_tokens: usize,
    lvl_first: usize,
    prefix_len: usize,
    raw_token_prefix: bool,
    contextual_prefix: bool,
    prefix_pad_id: Option<i64>,
    query_level0: bool,
    l0_ar: bool,
    use_rope: bool,

    rope_cos_gen: Vec<Tensor>,
    rope_sin_gen: Vec<Tensor>,
    wpe: Option<Embedding>,
    wp_prefix: Option<Tensor>,
    wp_1l: Option<Tensor>,
    lvl_1l: Tensor,

    wte: Linear,
    wle: Option<Embedding>,
    blocks: Vec<Block>,
    ln_f: LayerNorm,

    lm_heads: Vec<OutputHead>,
    lm_head_l0: Option<OutputHead>,
    l0_queries: Option<Tensor>,
    post_upsample_convs: Option<Vec<Conv1d>>,

    ctx_wte: Option<Embedding>,
    ctx_blocks: Vec<Block>,
    ctx_ln: Option<LayerNorm>,
}

impl NextConceptTransformer {
    pub fn load(vqvae: Arc<Vqvae>, cfg: TransformerConfig, vb: VarBuilder) -> anyhow::Result<Self> {
        let levels = vqvae.vq.levels.clone();
        let total_level_tokens: usize = levels.iter().sum();
        let query_level0 = cfg.query_level0;
        let l0_ar = cfg.l0_autoregressive && query_level0;
        let lvl_first = usize::from(!query_level0);
        let prefix_len = cfg.prefix_len;
        let dev = vb.device();

        let levels_target = &levels[lvl_first..];
        let mut rope_cos_gen = Vec::new();
        let mut rope_sin_gen = Vec::new();
        let mut wpe = None;
        let mut wp_prefix = None;
        let mut wp_1l = None;

        if cfg.use_rope {
            let head_dim = cfg.n_embd / cfg.n_head;
            for &l_k in levels_target {
                let mut pos_k: Vec<u32> = (0..prefix_len as u32).collect();
                pos_k.extend(0..l_k as u32);
                let (c, s) = precompute_rope_cos_sin(&pos_k, head_dim, cfg.rope_base, dev)?;
                rope_cos_gen.push(c);
                rope_sin_gen.push(s);
            }
        } else {
            let max_target = levels_target.iter().copied().max().unwrap_or(1);
            let t_max = prefix_len.max(max_target);
            wpe = Some(embedding(t_max, cfg.n_embd, vb.pp("wpe"))?);
            let p_idx: Vec<u32> = (0..prefix_len as u32).collect();
            wp_prefix = Some(Tensor::from_vec(p_idx, (1, prefix_len), dev)?);
            let mut t_idx = Vec::new();
            for &l in levels_target {
                t_idx.extend(0..l as u32);
            }
            let t_len = t_idx.len();
            wp_1l = Some(Tensor::from_vec(t_idx, (1, t_len), dev)?);
        }

        let mut lvl_ids = Vec::new();
        for (i, &l) in levels_target.iter().enumerate() {
            lvl_ids.extend(std::iter::repeat_n(i as u32, l));
        }
        let lvl_len = lvl_ids.len();
        let lvl_1l = Tensor::from_vec(lvl_ids, (1, lvl_len), dev)?;

        let t_vb = vb.pp("transformer");
        let wte = linear_with_bias(vqvae.embed_dim, cfg.n_embd, cfg.bias, t_vb.pp("wte"))?;
        let wle = if cfg.use_level_encoding {
            Some(embedding(levels_target.len(), cfg.n_embd, t_vb.pp("wle"))?)
        } else {
            None
        };

        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            blocks.push(Block::load(
                &cfg,
                AttentionMode::Bidirectional,
                t_vb.pp("blocks").pp(i),
            )?);
        }
        let ln_f = LayerNorm::load(cfg.n_embd, cfg.bias, t_vb.pp("ln_f"))?;

        let n_heads_out = cfg.vocab_size.len();
        let mut lm_heads = Vec::with_capacity(n_heads_out);
        let heads_vb = vb.pp("lm_heads");
        if cfg.shared_output_head {
            let hb = cfg.untied_head_bias && !cfg.shared_head_per_level_bias;
            let head = linear_with_bias(cfg.n_embd, cfg.vocab_size[0], hb, heads_vb.pp("head"))?;
            for i in 0..n_heads_out {
                let bias = if cfg.shared_head_per_level_bias {
                    Some(
                        heads_vb
                            .pp("biases")
                            .get(cfg.vocab_size[0], &i.to_string())?,
                    )
                } else {
                    None
                };
                lm_heads.push(OutputHead::Shared {
                    head: head.clone(),
                    bias,
                });
            }
        } else if cfg.tie_heads_to_codebook {
            for (i, &vs) in cfg.vocab_size.iter().enumerate() {
                let cb = vqvae.vq.embeddings[i + lvl_first].embeddings().clone();
                lm_heads.push(load_tied_head(cfg.n_embd, vs, cb, &cfg, heads_vb.pp(i))?);
            }
        } else {
            for (i, &vs) in cfg.vocab_size.iter().enumerate() {
                lm_heads.push(OutputHead::Linear(linear_with_bias(
                    cfg.n_embd,
                    vs,
                    cfg.untied_head_bias,
                    heads_vb.pp(i),
                )?));
            }
        }

        let lm_head_l0 = if cfg.predict_level0 {
            let v0 = vqvae.vq.vqvae_vocab_sizes[0];
            if cfg.tie_heads_to_codebook {
                let cb = vqvae.vq.embeddings[0].embeddings().clone();
                Some(load_tied_head(
                    cfg.n_embd,
                    v0,
                    cb,
                    &cfg,
                    vb.pp("lm_head_l0"),
                )?)
            } else {
                Some(OutputHead::Linear(linear_no_bias(
                    cfg.n_embd,
                    v0,
                    vb.pp("lm_head_l0"),
                )?))
            }
        } else {
            None
        };

        let l0_queries = if query_level0 {
            Some(vb.get((1, levels[0], vqvae.embed_dim), "l0_queries")?)
        } else {
            None
        };

        let post_upsample_convs = if cfg.post_upsample_conv.as_ref().is_some_and(|c| c.enabled) {
            let k = cfg.post_upsample_conv.as_ref().map_or(3, |c| c.kernel_size);
            let c_cfg = Conv1dConfig {
                padding: k / 2,
                ..Default::default()
            };
            let mut convs = Vec::with_capacity(levels_target.len());
            for i in 0..levels_target.len() {
                convs.push(conv1d(
                    vqvae.embed_dim,
                    vqvae.embed_dim,
                    k,
                    c_cfg,
                    vb.pp("post_upsample_convs").pp(i),
                )?);
            }
            Some(convs)
        } else {
            None
        };

        let mut ctx_wte = None;
        let mut ctx_blocks = Vec::new();
        let mut ctx_ln = None;
        if let Some(ctx_cfg) = &cfg.ctx_encoder
            && ctx_cfg.enabled
        {
            let ctx_vocab = ctx_cfg.vocab_size.unwrap_or(vqvae.in_vocab_size);
            ctx_wte = Some(embedding(ctx_vocab, cfg.n_embd, vb.pp("ctx_wte"))?);
            for i in 0..ctx_cfg.n_layer {
                ctx_blocks.push(Block::load(
                    &cfg,
                    AttentionMode::Bidirectional,
                    vb.pp("ctx_blocks").pp(i),
                )?);
            }
            if ctx_cfg.n_layer > 0 {
                ctx_ln = Some(LayerNorm::load(cfg.n_embd, cfg.bias, vb.pp("ctx_ln"))?);
            }
        }

        Ok(Self {
            raw_token_prefix: cfg.raw_token_prefix,
            contextual_prefix: cfg.contextual_prefix,
            prefix_pad_id: cfg.prefix_pad_token_id,
            use_rope: cfg.use_rope,
            config: cfg,
            vqvae,
            levels,
            total_level_tokens,
            lvl_first,
            prefix_len,
            query_level0,
            l0_ar,
            rope_cos_gen,
            rope_sin_gen,
            wpe,
            wp_prefix,
            wp_1l,
            lvl_1l,
            wte,
            wle,
            blocks,
            ln_f,
            lm_heads,
            lm_head_l0,
            l0_queries,
            post_upsample_convs,
            ctx_wte,
            ctx_blocks,
            ctx_ln,
        })
    }

    fn apply_post_upsample_conv(&self, chunk: &Tensor, hidx: usize) -> Result<Tensor> {
        let Some(convs) = &self.post_upsample_convs else {
            return Ok(chunk.clone());
        };
        let delta = convs[hidx]
            .forward(&chunk.transpose(1, 2)?)?
            .transpose(1, 2)?;
        chunk + delta
    }

    fn encode_context(&self, prefix_ids: &Tensor) -> Result<Tensor> {
        let wte = self
            .ctx_wte
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("ctx_wte not loaded".to_string()))?;
        let mut h = wte.forward(prefix_ids)?;
        let rope_p = if self.use_rope {
            Some((
                self.rope_cos_gen[0].narrow(0, 0, self.prefix_len)?,
                self.rope_sin_gen[0].narrow(0, 0, self.prefix_len)?,
            ))
        } else {
            None
        };
        for blk in &self.ctx_blocks {
            let (out, _) = blk.forward(&h, None, rope_p.as_ref(), None, false)?;
            h = out;
        }
        match &self.ctx_ln {
            Some(ln) => ln.forward(&h),
            None => Ok(h),
        }
    }

    fn kv_prefix(
        &self,
        mut h: Tensor,
        mask: Option<&Tensor>,
        rope: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let mut kv = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            let (out, layer_kv) = blk.forward(&h, mask, rope, None, true)?;
            h = out;
            kv.push(layer_kv.expect("return_kv=true must return KV"));
        }
        Ok((h, kv))
    }

    fn kv_level(
        &self,
        mut x: Tensor,
        prefix_kv: &[(Tensor, Tensor)],
        mask: Option<&Tensor>,
        rope: Option<&(Tensor, Tensor)>,
        hidx: usize,
    ) -> Result<Tensor> {
        for (blk, layer_kv) in self.blocks.iter().zip(prefix_kv.iter()) {
            let (out, _) = blk.forward(&x, mask, rope, Some(layer_kv), false)?;
            x = out;
        }
        let h = self.ln_f.forward(&x)?;
        self.lm_heads[hidx].forward(&h)
    }

    fn level0_slice(&self, x: &Tensor, pad_mask: Option<&Tensor>) -> Result<Tensor> {
        let p = self.prefix_len;
        let l0 = self.levels[0];
        let Some(pm) = pad_mask else {
            return x.narrow(1, p - l0, l0);
        };
        let (b, _, d) = x.dims3()?;
        let pm_u8 = pm.to_vec2::<u8>()?;
        let mut slices = Vec::with_capacity(b);
        for (bi, row) in pm_u8.iter().enumerate() {
            let non_pad = row.iter().copied().filter(|&v| v == 0).count();
            let last = non_pad.saturating_sub(1).max(l0 - 1).min(p - 1);
            let start = last + 1 - l0;
            slices.push(x.narrow(0, bi, 1)?.narrow(1, start, l0)?);
        }
        Tensor::cat(&slices, 0)?.reshape((b, l0, d))
    }

    pub fn generate(
        &self,
        initial_codes: Option<&Tensor>,
        prefix_tokens: &Tensor,
        opts: &GenerateOptions,
        rng: &mut ChaCha8Rng,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let dev = prefix_tokens.device();
        let mut prefix_tokens = prefix_tokens.clone();
        let mut idx = initial_codes.cloned();

        let cfg_b = if opts.cfg_scale > 0.0 {
            let pad_id = self
                .prefix_pad_id
                .ok_or_else(|| anyhow::anyhow!("cfg_scale needs prefix_pad_token_id"))?;
            let (b, p_len) = prefix_tokens.dims2()?;
            let mut null_vec = vec![pad_id as u32; b * p_len];
            for row in 0..b {
                null_vec[row * p_len] = opts.cfg_null_eos;
            }
            let null_t = Tensor::from_vec(null_vec, (b, p_len), dev)?;
            prefix_tokens = Tensor::cat(&[&prefix_tokens, &null_t], 0)?;
            if let Some(ref ic) = idx {
                idx = Some(Tensor::cat(&[ic, ic], 0)?);
            }
            Some(b)
        } else {
            None
        };

        let b = prefix_tokens.dim(0)?;
        let p = self.prefix_len;
        let full_len = *self.levels.last().unwrap_or(&1);
        let mut z_hat = Tensor::zeros((b, full_len, self.vqvae.embed_dim), DType::F32, dev)?;
        let mut trajectory = Vec::new();

        let mut prefix_emb = if self.ctx_wte.is_some() {
            let p_slice = prefix_tokens.narrow(1, 0, p)?;
            self.encode_context(&p_slice)?
        } else {
            let z_prefix = if self.raw_token_prefix {
                if self.contextual_prefix && self.vqvae.encoder.has_blocks() {
                    self.vqvae
                        .encoder
                        .forward_nopos(&prefix_tokens)?
                        .narrow(1, 0, p)?
                } else {
                    self.vqvae
                        .encoder
                        .wte
                        .forward(&prefix_tokens)?
                        .narrow(1, 0, p)?
                }
            } else {
                self.vqvae.encode_z(&prefix_tokens)?.narrow(1, 0, p)?
            };
            self.wte
                .forward(&z_prefix.to_dtype(self.wte.weight().dtype())?)?
        };

        if let (Some(wpe), Some(wp_p)) = (&self.wpe, &self.wp_prefix) {
            prefix_emb = prefix_emb.broadcast_add(&wpe.forward(wp_p)?)?;
        }

        let mut level_pos_embed = match &self.wle {
            Some(wle) => wle.forward(&self.lvl_1l)?,
            None => Tensor::zeros(
                (1, self.lvl_1l.dim(1)?, self.config.n_embd),
                prefix_emb.dtype(),
                dev,
            )?,
        };
        if let (Some(wpe), Some(wp_1l)) = (&self.wpe, &self.wp_1l) {
            level_pos_embed = level_pos_embed.broadcast_add(&wpe.forward(wp_1l)?)?;
        }

        let (pad_mask_u8, pbias) = if let Some(pad_id) = self.prefix_pad_id {
            let p_slice = prefix_tokens.narrow(1, 0, p)?;
            let rows = p_slice.to_vec2::<u32>()?;
            let mut mask_u8 = vec![0u8; b * p];
            let mut bias_f = vec![0.0f32; b * p];
            for (bi, row) in rows.iter().enumerate() {
                for (pi, &tok) in row.iter().enumerate() {
                    if i64::from(tok) == pad_id {
                        mask_u8[bi * p + pi] = 1;
                        bias_f[bi * p + pi] = f32::NEG_INFINITY;
                    }
                }
            }
            let pm = Tensor::from_vec(mask_u8, (b, p), dev)?;
            let pb = Tensor::from_vec(bias_f, (b, 1, 1, p), dev)?;
            (Some(pm), Some(pb))
        } else {
            (None, None)
        };

        let mut prefix_kv: Option<Vec<(Tensor, Tensor)>> = None;
        let mut soft_embedding: Option<Tensor> = None;

        if idx.is_none() && self.query_level0 {
            let l0 = self.levels[0];
            let q0_base = self
                .l0_queries
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing l0_queries"))?
                .expand((b, l0, self.vqvae.embed_dim))?
                .to_dtype(prefix_emb.dtype())?;
            let q0 = self.apply_post_upsample_conv(&q0_base, 0)?;
            let mut q = self.wte.forward(&q0)?;
            if let (Some(wpe), Some(wp_1l)) = (&self.wpe, &self.wp_1l) {
                q = q.broadcast_add(&wpe.forward(&wp_1l.narrow(1, 0, l0)?)?)?;
            }
            if let Some(wle) = &self.wle {
                q = q.broadcast_add(&wle.forward(&self.lvl_1l.narrow(1, 0, l0)?)?)?;
            }

            let mut m0 = build_l0_mask(p, l0, self.l0_ar, dev)?;
            if let Some(ref pb) = pbias {
                let pb_padded = pad_last_dim(pb, l0, 0.0)?;
                m0 = m0.broadcast_add(&pb_padded)?;
            }
            let rope_0 = if self.use_rope {
                Some((self.rope_cos_gen[0].clone(), self.rope_sin_gen[0].clone()))
            } else {
                None
            };

            if self.l0_ar {
                let mut codes = Vec::with_capacity(l0);
                let mut soft_codes = Vec::with_capacity(l0);
                let mut x_in = q0_base.clone();
                for t in 0..l0 {
                    let mut qt = self.wte.forward(&x_in)?;
                    if let (Some(wpe), Some(wp_1l)) = (&self.wpe, &self.wp_1l) {
                        qt = qt.broadcast_add(&wpe.forward(&wp_1l.narrow(1, 0, l0)?)?)?;
                    }
                    if let Some(wle) = &self.wle {
                        qt = qt.broadcast_add(&wle.forward(&self.lvl_1l.narrow(1, 0, l0)?)?)?;
                    }
                    let logits = if opts.use_kv_cache
                        && let Some(ref pkv) = prefix_kv
                    {
                        let m_sub = m0.narrow(2, p, l0)?;
                        self.kv_level(qt, pkv, Some(&m_sub), rope_0.as_ref(), 0)?
                    } else {
                        let mut h = Tensor::cat(&[&prefix_emb, &qt], 1)?;
                        if opts.use_kv_cache {
                            let (h_out, layer_kv) =
                                self.kv_prefix(h, Some(&m0), rope_0.as_ref())?;
                            h = h_out;
                            prefix_kv = Some(
                                layer_kv
                                    .into_iter()
                                    .map(|(k, v)| {
                                        Ok((
                                            k.narrow(2, 0, p)?.contiguous()?,
                                            v.narrow(2, 0, p)?.contiguous()?,
                                        ))
                                    })
                                    .collect::<Result<Vec<_>>>()?,
                            );
                        } else {
                            for blk in &self.blocks {
                                let (out, _) =
                                    blk.forward(&h, Some(&m0), rope_0.as_ref(), None, false)?;
                                h = out;
                            }
                        }
                        let h_t = self.ln_f.forward(&h)?.narrow(1, p, l0)?;
                        self.lm_heads[0].forward(&h_t)?
                    };
                    let step_logits = logits.narrow(1, t, 1)?;
                    let guided =
                        apply_cfg(&step_logits, cfg_b, opts.cfg_scale, 0, self.levels.len())?;
                    let tok = sample_level(&guided, opts.temperature, opts.top_k, opts.top_p, rng)?;
                    let tok = if cfg_b.is_some() {
                        Tensor::cat(&[&tok, &tok], 0)?
                    } else {
                        tok
                    };
                    codes.push(tok);
                    if opts.soft_decisions {
                        soft_codes.push(soft_code_embedding(
                            &guided,
                            &self.vqvae.vq.embeddings[0],
                            opts.soft_temperature,
                            cfg_b.is_some(),
                        )?);
                    }
                    if t + 1 < l0 {
                        let known_emb = if opts.soft_decisions {
                            Tensor::cat(&soft_codes, 1)?
                        } else {
                            let known = Tensor::cat(&codes, 1)?;
                            self.vqvae.vq.embeddings[0].forward(&known)?
                        };
                        let mut parts = vec![
                            q0_base.narrow(1, 0, 1)?,
                            known_emb.to_dtype(q0_base.dtype())?,
                        ];
                        if t + 2 < l0 {
                            parts.push(q0_base.narrow(1, t + 2, l0 - (t + 2))?);
                        }
                        x_in = Tensor::cat(&parts, 1)?;
                    }
                }
                idx = Some(Tensor::cat(&codes, 1)?);
                if opts.soft_decisions {
                    soft_embedding = Some(Tensor::cat(&soft_codes, 1)?);
                }
            } else {
                let mut h = Tensor::cat(&[&prefix_emb, &q], 1)?;
                if opts.use_kv_cache {
                    let (h_out, layer_kv) = self.kv_prefix(h, Some(&m0), rope_0.as_ref())?;
                    h = h_out;
                    prefix_kv = Some(
                        layer_kv
                            .into_iter()
                            .map(|(k, v)| {
                                Ok((
                                    k.narrow(2, 0, p)?.contiguous()?,
                                    v.narrow(2, 0, p)?.contiguous()?,
                                ))
                            })
                            .collect::<Result<Vec<_>>>()?,
                    );
                } else {
                    for blk in &self.blocks {
                        let (out, _) = blk.forward(&h, Some(&m0), rope_0.as_ref(), None, false)?;
                        h = out;
                    }
                }
                let h = self.ln_f.forward(&h)?.narrow(1, p, l0)?;
                let logits = self.lm_heads[0].forward(&h)?;
                let guided = apply_cfg(&logits, cfg_b, opts.cfg_scale, 0, self.levels.len())?;
                let tok = sample_level(&guided, opts.temperature, opts.top_k, opts.top_p, rng)?;
                idx = Some(if cfg_b.is_some() {
                    Tensor::cat(&[&tok, &tok], 0)?
                } else {
                    tok
                });
                if opts.soft_decisions {
                    soft_embedding = Some(soft_code_embedding(
                        &guided,
                        &self.vqvae.vq.embeddings[0],
                        opts.soft_temperature,
                        cfg_b.is_some(),
                    )?);
                }
            }
        }

        if idx.is_none() {
            let head_l0 = self.lm_head_l0.as_ref().ok_or_else(|| {
                anyhow::anyhow!("generate(idx=None) requires predict_level0 or query_level0")
            })?;
            let rope_p = if self.use_rope {
                Some((
                    self.rope_cos_gen[0].narrow(0, 0, p)?,
                    self.rope_sin_gen[0].narrow(0, 0, p)?,
                ))
            } else {
                None
            };
            let mut pfx_mask = Tensor::zeros((1, 1, p, p), DType::F32, dev)?;
            if let Some(ref pb) = pbias {
                pfx_mask = pfx_mask.broadcast_add(pb)?;
            }
            let mut h = prefix_emb.clone();
            if opts.use_kv_cache {
                let (h_out, pkv) = self.kv_prefix(h, Some(&pfx_mask), rope_p.as_ref())?;
                h = h_out;
                prefix_kv = Some(pkv);
            } else {
                for blk in &self.blocks {
                    let (out, _) =
                        blk.forward(&h, Some(&pfx_mask), rope_p.as_ref(), None, false)?;
                    h = out;
                }
            }
            let h = self.ln_f.forward(&h)?;
            let l0_h = self.level0_slice(&h, pad_mask_u8.as_ref())?;
            let l0_logits = head_l0.forward(&l0_h)?;
            let guided = apply_cfg(&l0_logits, cfg_b, opts.cfg_scale, 0, self.levels.len())?;
            let tok = sample_level(&guided, opts.temperature, opts.top_k, opts.top_p, rng)?;
            idx = Some(if cfg_b.is_some() {
                Tensor::cat(&[&tok, &tok], 0)?
            } else {
                tok
            });
            if opts.soft_decisions {
                soft_embedding = Some(soft_code_embedding(
                    &guided,
                    &self.vqvae.vq.embeddings[0],
                    opts.soft_temperature,
                    cfg_b.is_some(),
                )?);
            }
        }

        let mut idx_next = idx.expect("idx initialized");
        let mut generated_idx = vec![idx_next.clone()];

        if opts.use_kv_cache && prefix_kv.is_none() {
            let rope_p = if self.use_rope {
                Some((
                    self.rope_cos_gen[0].narrow(0, 0, p)?,
                    self.rope_sin_gen[0].narrow(0, 0, p)?,
                ))
            } else {
                None
            };
            let mut pfx_mask = Tensor::zeros((1, 1, p, p), DType::F32, dev)?;
            if let Some(ref pb) = pbias {
                pfx_mask = pfx_mask.broadcast_add(pb)?;
            }
            let (_, pkv) = self.kv_prefix(prefix_emb.clone(), Some(&pfx_mask), rope_p.as_ref())?;
            prefix_kv = Some(pkv);
        }

        let mut level_acum = if self.query_level0 { self.levels[0] } else { 0 };
        let n_levels = self.levels.len();

        for level_idx in 0..(n_levels - 1) {
            let z_q_level = match soft_embedding.take() {
                Some(se) => se,
                None => self.vqvae.vq.embeddings[level_idx]
                    .forward(&idx_next)?
                    .to_dtype(DType::F32)?,
            };
            let (new_z_hat, mut x) = self
                .vqvae
                .vq
                .get_next_autoregressive_input(level_idx, &z_hat, &z_q_level)?;
            z_hat = new_z_hat;
            if opts.return_trajectory {
                trajectory.push(z_hat.clone());
            }

            let hidx = level_idx + (1 - self.lvl_first);
            x = self.apply_post_upsample_conv(&x, hidx)?;

            let l_cur = self.levels[level_idx + 1];
            let pos_slice = level_pos_embed.narrow(1, level_acum, l_cur)?;
            x = self
                .wte
                .forward(&x.to_dtype(self.wte.weight().dtype())?)?
                .broadcast_add(&pos_slice)?;

            let rope_k = if self.use_rope {
                Some((
                    self.rope_cos_gen[hidx].clone(),
                    self.rope_sin_gen[hidx].clone(),
                ))
            } else {
                None
            };

            let logits = if opts.use_kv_cache {
                let pkv = prefix_kv.as_ref().expect("prefix_kv built");
                let mut lvl_mask = Tensor::zeros((1, 1, l_cur, p + l_cur), DType::F32, dev)?;
                if let Some(ref pb) = pbias {
                    lvl_mask = lvl_mask.broadcast_add(&pad_last_dim(pb, l_cur, 0.0)?)?;
                }
                self.kv_level(x, pkv, Some(&lvl_mask), rope_k.as_ref(), hidx)?
            } else {
                let mut gen_mask = build_l0_mask(p, l_cur, false, dev)?;
                if let Some(ref pb) = pbias {
                    gen_mask = gen_mask.broadcast_add(&pad_last_dim(pb, l_cur, 0.0)?)?;
                }
                let mut h = Tensor::cat(&[&prefix_emb, &x], 1)?;
                for blk in &self.blocks {
                    let (out, _) =
                        blk.forward(&h, Some(&gen_mask), rope_k.as_ref(), None, false)?;
                    h = out;
                }
                let h = self.ln_f.forward(&h)?.narrow(1, p, l_cur)?;
                self.lm_heads[hidx].forward(&h)?
            };

            let guided = apply_cfg(&logits, cfg_b, opts.cfg_scale, level_idx + 1, n_levels)?;
            let tok = sample_level(&guided, opts.temperature, opts.top_k, opts.top_p, rng)?;
            idx_next = if cfg_b.is_some() {
                Tensor::cat(&[&tok, &tok], 0)?
            } else {
                tok
            };
            if opts.soft_decisions {
                soft_embedding = Some(soft_code_embedding(
                    &guided,
                    &self.vqvae.vq.embeddings[level_idx + 1],
                    opts.soft_temperature,
                    cfg_b.is_some(),
                )?);
            }
            generated_idx.push(idx_next.clone());
            level_acum += l_cur;
        }

        let last = n_levels - 1;
        let z_q_last = match soft_embedding.take() {
            Some(se) => se,
            None => self.vqvae.vq.embeddings[last]
                .forward(&idx_next)?
                .to_dtype(DType::F32)?,
        };
        let (new_z_hat, _) = self
            .vqvae
            .vq
            .get_next_autoregressive_input(last, &z_hat, &z_q_last)?;
        z_hat = new_z_hat;
        if opts.return_trajectory {
            trajectory.push(z_hat.clone());
            z_hat = Tensor::stack(&trajectory, 1)?;
        }

        let mut out = Tensor::cat(&generated_idx, 1)?;
        if let Some(b_orig) = cfg_b {
            out = out.narrow(0, 0, b_orig)?;
            z_hat = z_hat.narrow(0, 0, b_orig)?;
        }
        Ok((out, z_hat))
    }
}

fn load_tied_head(
    n_embd: usize,
    vocab_size: usize,
    codebook: Tensor,
    cfg: &TransformerConfig,
    vb: VarBuilder,
) -> Result<OutputHead> {
    let d = codebook.dim(1)?;
    let proj = linear_no_bias(n_embd, d, vb.pp("proj"))?;
    let bias = if cfg.tie_heads_bias {
        Some(vb.get(vocab_size, "bias")?)
    } else {
        None
    };
    let e_tensor = if cfg.tie_heads_trainable {
        vb.get((vocab_size, d), "E")?
    } else {
        codebook
    };
    let e_f32 = e_tensor.to_dtype(DType::F32)?;
    let e_sq = e_f32.sqr()?.sum(D::Minus1)?;
    let d_scale = f64::from(e_sq.mean_all()?.to_scalar::<f32>()?.max(1e-12));
    let log_inv_t = if cfg.tie_heads_learn_scale {
        f64::from(
            vb.get((), "log_inv_T")?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?,
        )
    } else {
        0.0
    };
    Ok(OutputHead::TiedCodebook {
        proj,
        bias,
        codebook: e_tensor,
        e_sq,
        d_scale,
        log_inv_t,
    })
}

fn build_l0_mask(p: usize, l0: usize, causal_l0: bool, device: &Device) -> Result<Tensor> {
    let s = p + l0;
    let mut data = vec![0.0f32; s * s];
    for i in 0..p {
        for j in p..s {
            data[i * s + j] = f32::NEG_INFINITY;
        }
    }
    if causal_l0 {
        for i in 0..l0 {
            for j in (i + 1)..l0 {
                data[(p + i) * s + (p + j)] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, s, s), device)
}

fn pad_last_dim(t: &Tensor, extra: usize, val: f32) -> Result<Tensor> {
    let (d0, d1, d2, _) = t.dims4()?;
    let pad = Tensor::full(val, (d0, d1, d2, extra), t.device())?.to_dtype(t.dtype())?;
    Tensor::cat(&[t, &pad], D::Minus1)
}

fn apply_cfg(
    logits: &Tensor,
    cfg_b: Option<usize>,
    cfg_scale: f64,
    level: usize,
    nlev: usize,
) -> Result<Tensor> {
    let Some(b) = cfg_b else {
        return Ok(logits.clone());
    };
    if cfg_scale <= 0.0 {
        return Ok(logits.clone());
    }
    let t = cfg_scale * (level as f64 / nlev.saturating_sub(1).max(1) as f64);
    let cond = logits.narrow(0, 0, b)?;
    let uncond = logits.narrow(0, b, b)?;
    (cond * (1.0 + t))? - (uncond * t)?
}

fn soft_code_embedding(
    guided_logits: &Tensor,
    emb: &Embedding,
    soft_temp: f64,
    double_for_cfg: bool,
) -> Result<Tensor> {
    let scaled = (guided_logits.to_dtype(DType::F32)? / soft_temp)?;
    let probs = ops::softmax(&scaled, D::Minus1)?;
    let w = emb.embeddings().to_dtype(DType::F32)?;
    let e = probs.broadcast_matmul(&w)?;
    if double_for_cfg {
        Tensor::cat(&[&e, &e], 0)
    } else {
        Ok(e)
    }
}

fn sample_level(
    logits: &Tensor,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    rng: &mut ChaCha8Rng,
) -> Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    if temperature == 0.0 {
        return logits.argmax(D::Minus1);
    }
    let flat = (logits.to_dtype(DType::F32)? / temperature)?.reshape((b * t, v))?;
    let rows = flat.to_vec2::<f32>()?;
    let mut picked = Vec::with_capacity(b * t);

    for mut row in rows {
        if top_k > 0 && top_k < v {
            let mut sorted = row.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let kth = sorted[top_k - 1];
            for val in &mut row {
                if *val < kth {
                    *val = f32::NEG_INFINITY;
                }
            }
        }
        if top_p > 0.0 && top_p < 1.0 {
            let mut indexed: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
            let max_v = indexed[0].1;
            let exps: Vec<f32> = indexed.iter().map(|&(_, x)| (x - max_v).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let mut cum = 0.0f32;
            for (rank, &(idx, _)) in indexed.iter().enumerate() {
                let p = exps[rank] / sum;
                if cum > top_p as f32 {
                    row[idx] = f32::NEG_INFINITY;
                }
                cum += p;
            }
        }
        let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let probs: Vec<f64> = row
            .iter()
            .map(|&x| {
                if x.is_finite() {
                    f64::from(x - max_v).exp()
                } else {
                    0.0
                }
            })
            .collect();
        let total: f64 = probs.iter().sum();
        let mut draw = rng.random::<f64>() * total;
        let mut chosen = 0u32;
        for (i, &p) in probs.iter().enumerate() {
            draw -= p;
            if draw <= 0.0 {
                chosen = i as u32;
                break;
            }
        }
        picked.push(chosen);
    }

    Tensor::from_vec(picked, (b, t), logits.device())
}
