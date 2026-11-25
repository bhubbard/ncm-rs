use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Embedding, Linear, Module,
    VarBuilder, conv_transpose1d, conv1d, embedding, linear_no_bias,
};

use crate::config::{TransformerConfig, VQVAEConfig};
use crate::models::common::{AttentionMode, Block, LayerNorm};

fn vae_block_config(embed_dim: usize, n_head: usize, use_qk_norm: bool) -> TransformerConfig {
    TransformerConfig {
        n_layer: 1,
        n_head,
        n_kv_heads: None,
        bias: true,
        dropout: 0.0,
        n_embd: embed_dim,
        context_length: 0,
        prefix_len: 0,
        raw_token_prefix: false,
        contextual_prefix: false,
        prefix_pad_token_id: None,
        vocab_size: Vec::new(),
        predict_level0: false,
        attn_pattern: "block_diagonal".to_string(),
        use_positional_encoding: false,
        use_level_encoding: false,
        use_rope: false,
        rope_base: 10_000,
        use_qk_norm,
        use_swiglu: false,
        use_geglu: false,
        use_relu2: false,
        mlp_ratio: 4.0,
        parallel_block: false,
        normformer: false,
        use_layerscale: false,
        layerscale_init: 1e-4,
        use_rezero: false,
        ctx_encoder: None,
        post_upsample_conv: None,
        use_flex_attention: false,
        query_level0: false,
        l0_autoregressive: false,
        shared_output_head: false,
        shared_head_per_level_bias: false,
        untied_head_bias: true,
        tie_heads_to_codebook: false,
        tie_heads_learn_scale: true,
        tie_heads_trainable: false,
        tie_heads_bias: false,
    }
}

#[derive(Debug, Clone)]
struct ConvResBlock {
    ln: LayerNorm,
    c1: Conv1d,
    c2: Conv1d,
}

impl ConvResBlock {
    fn load(embed_dim: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: kernel_size / 2,
            ..Default::default()
        };
        Ok(Self {
            ln: LayerNorm::load(embed_dim, false, vb.pp("ln"))?,
            c1: conv1d(embed_dim, embed_dim, kernel_size, cfg, vb.pp("c1"))?,
            c2: conv1d(embed_dim, embed_dim, kernel_size, cfg, vb.pp("c2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.ln.forward(x)?.transpose(1, 2)?;
        let h = self.c1.forward(&h)?.gelu_erf()?;
        let h = self.c2.forward(&h)?.transpose(1, 2)?;
        x + h
    }
}

#[derive(Debug, Clone)]
pub struct Encoder {
    pub wte: Embedding,
    wpe: Option<Embedding>,
    blocks: Vec<Block>,
    conv_blocks: Vec<ConvResBlock>,
    ln_e: Option<LayerNorm>,
    down: Option<Conv1d>,
    embed_dim: usize,
}

impl Encoder {
    pub fn load(cfg: &VQVAEConfig, vb: VarBuilder) -> Result<Self> {
        let wte = embedding(cfg.in_vocab_size, cfg.embed_dim, vb.pp("wte"))?;
        let wpe = if cfg.enc_positional {
            Some(embedding(cfg.context_length, cfg.embed_dim, vb.pp("wpe"))?)
        } else {
            None
        };

        let block_cfg = vae_block_config(cfg.embed_dim, cfg.n_layers, cfg.use_qk_norm);
        let mode = if cfg.enc_attn_window > 0 {
            AttentionMode::Window(cfg.enc_attn_window)
        } else {
            AttentionMode::Bidirectional
        };
        let mut blocks = Vec::with_capacity(cfg.enc_layers);
        for i in 0..cfg.enc_layers {
            blocks.push(Block::load(&block_cfg, mode, vb.pp("blocks").pp(i))?);
        }

        let mut conv_blocks = Vec::with_capacity(cfg.enc_conv_layers);
        for i in 0..cfg.enc_conv_layers {
            conv_blocks.push(ConvResBlock::load(
                cfg.embed_dim,
                cfg.enc_conv_kernel,
                vb.pp("conv_blocks").pp(i),
            )?);
        }

        let ln_e = if cfg.enc_layers > 0 || cfg.enc_conv_layers > 0 {
            Some(LayerNorm::load(cfg.embed_dim, false, vb.pp("ln_e"))?)
        } else {
            None
        };

        let down = if cfg.compression_factor > 1 {
            let f = cfg.compression_factor;
            let c_cfg = Conv1dConfig {
                stride: f,
                ..Default::default()
            };
            Some(conv1d(
                cfg.embed_dim,
                cfg.embed_dim,
                f,
                c_cfg,
                vb.pp("down"),
            )?)
        } else {
            None
        };

        Ok(Self {
            wte,
            wpe,
            blocks,
            conv_blocks,
            ln_e,
            down,
            embed_dim: cfg.embed_dim,
        })
    }

    pub fn has_blocks(&self) -> bool {
        !self.blocks.is_empty()
    }

    fn run_stacks(&self, mut x: Tensor) -> Result<Tensor> {
        for blk in &self.conv_blocks {
            x = blk.forward(&x)?;
        }
        for blk in &self.blocks {
            let (out, _) = blk.forward(&x, None, None, None, false)?;
            x = out;
        }
        if let Some(ln) = &self.ln_e {
            x = ln.forward(&x)?;
        }
        Ok(x)
    }

    pub fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let mut x = self.wte.forward(idx)?;
        if let Some(wpe) = &self.wpe {
            let t = idx.dim(1)?;
            let pos = Tensor::arange(0u32, t as u32, idx.device())?;
            x = x.broadcast_add(&wpe.forward(&pos)?)?;
        }
        x = self.run_stacks(x)?;
        if let Some(down) = &self.down {
            x = down.forward(&x.transpose(1, 2)?)?.transpose(1, 2)?;
        }
        Ok(x)
    }

    pub fn forward_nopos(&self, idx: &Tensor) -> Result<Tensor> {
        let t = idx.dim(1)?;
        let tok = self.wte.forward(idx)?;
        let pe = sinusoidal_pe(t, self.embed_dim, idx.device(), tok.dtype())?;
        let x = tok.broadcast_add(&pe.unsqueeze(0)?)?;
        self.run_stacks(x)
    }
}

fn sinusoidal_pe(t: usize, c: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let mut data = vec![0.0f32; t * c];
    for pos in 0..t {
        for i in (0..c).step_by(2) {
            let div = (-(10000.0f64.ln()) * (i as f64) / (c as f64)).exp();
            let angle = (pos as f64) * div;
            data[pos * c + i] = angle.sin() as f32;
            if i + 1 < c {
                data[pos * c + i + 1] = angle.cos() as f32;
            }
        }
    }
    Tensor::from_vec(data, (t, c), device)?.to_dtype(dtype)
}

#[derive(Debug, Clone)]
pub struct Decoder {
    up: Option<ConvTranspose1d>,
    wpe: Option<Embedding>,
    middle_block: Block,
    extra_blocks: Vec<Block>,
    ln_f: LayerNorm,
    output_layer: Linear,
}

impl Decoder {
    pub fn load(cfg: &VQVAEConfig, vb: VarBuilder) -> Result<Self> {
        let up = if cfg.compression_factor > 1 {
            let f = cfg.compression_factor;
            let ct_cfg = ConvTranspose1dConfig {
                stride: f,
                ..Default::default()
            };
            Some(conv_transpose1d(
                cfg.embed_dim,
                cfg.embed_dim,
                f,
                ct_cfg,
                vb.pp("up"),
            )?)
        } else {
            None
        };

        let wpe = if cfg.dec_positional {
            Some(embedding(cfg.context_length, cfg.embed_dim, vb.pp("wpe"))?)
        } else {
            None
        };

        let block_cfg = vae_block_config(cfg.embed_dim, cfg.n_layers, cfg.use_qk_norm);
        let middle_block = Block::load(
            &block_cfg,
            AttentionMode::Bidirectional,
            vb.pp("middle_block"),
        )?;
        let n_extra = cfg.dec_layers.saturating_sub(1);
        let mut extra_blocks = Vec::with_capacity(n_extra);
        for i in 0..n_extra {
            extra_blocks.push(Block::load(
                &block_cfg,
                AttentionMode::Bidirectional,
                vb.pp("extra_blocks").pp(i),
            )?);
        }

        let ln_f = LayerNorm::load(cfg.embed_dim, false, vb.pp("ln_f"))?;
        let output_layer = linear_no_bias(cfg.embed_dim, cfg.in_vocab_size, vb.pp("output_layer"))?;

        Ok(Self {
            up,
            wpe,
            middle_block,
            extra_blocks,
            ln_f,
            output_layer,
        })
    }

    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let mut z = z.clone();
        if let Some(up) = &self.up {
            z = up.forward(&z.transpose(1, 2)?)?.transpose(1, 2)?;
        }
        if let Some(wpe) = &self.wpe {
            let t = z.dim(1)?;
            let pos = Tensor::arange(0u32, t as u32, z.device())?;
            z = z.broadcast_add(&wpe.forward(&pos)?)?;
        }
        let (mut x, _) = self.middle_block.forward(&z, None, None, None, false)?;
        for blk in &self.extra_blocks {
            let (out, _) = blk.forward(&x, None, None, None, false)?;
            x = out;
        }
        let x = self.ln_f.forward(&x)?;
        self.output_layer.forward(&x)
    }
}
