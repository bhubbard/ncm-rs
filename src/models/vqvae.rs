use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::VQVAEConfig;
use crate::models::quant::{MultiscaleResidualVectorQuantizer, group_norm_no_affine};
use crate::models::vae::{Decoder, Encoder};

#[derive(Debug, Clone)]
pub struct Vqvae {
    pub embed_dim: usize,
    pub in_vocab_size: usize,
    pub context_length: usize,
    pub encoder: Encoder,
    pub vq: MultiscaleResidualVectorQuantizer,
    pub decoder: Decoder,
    pre_quant_norm: PreQuantNorm,
}

#[derive(Debug, Clone)]
enum PreQuantNorm {
    None,
    Layer,
    Group(usize),
}

impl Vqvae {
    pub fn load(cfg: &VQVAEConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::load(cfg, vb.pp("encoder"))?;
        let vq = MultiscaleResidualVectorQuantizer::load(
            &cfg.vqvae_vocab_size,
            cfg.embed_dim,
            &cfg.vector_quantizer_config,
            vb.pp("vq"),
        )?;
        let decoder = Decoder::load(cfg, vb.pp("decoder"))?;

        let pre_quant_norm = match cfg.pre_quant_groupnorm {
            None | Some(0) => PreQuantNorm::None,
            Some(g) => {
                if cfg.pre_quant_norm_type == "layer" {
                    PreQuantNorm::Layer
                } else {
                    PreQuantNorm::Group(g)
                }
            }
        };

        Ok(Self {
            embed_dim: cfg.embed_dim,
            in_vocab_size: cfg.in_vocab_size,
            context_length: cfg.context_length,
            encoder,
            vq,
            decoder,
            pre_quant_norm,
        })
    }

    pub fn encode_z(&self, x: &Tensor) -> Result<Tensor> {
        let z = self.encoder.forward(x)?;
        match self.pre_quant_norm {
            PreQuantNorm::None => Ok(z),
            PreQuantNorm::Layer => {
                let orig = z.dtype();
                let zf = z.to_dtype(DType::F32)?;
                let mean = zf.mean_keepdim(candle_core::D::Minus1)?;
                let diff = zf.broadcast_sub(&mean)?;
                let var = diff.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
                let inv_std = (var + 1e-5)?.sqrt()?.recip()?;
                diff.broadcast_mul(&inv_std)?.to_dtype(orig)
            }
            PreQuantNorm::Group(g) => {
                let zt = z.transpose(1, 2)?;
                group_norm_no_affine(&zt, g, 1e-5)?.transpose(1, 2)
            }
        }
    }

    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let z = self.encode_z(x)?;
        self.vq.z_to_idx(&z)
    }

    pub fn decode(&self, idx: &Tensor) -> Result<Tensor> {
        let z_q = self.vq.idx_to_z_q(idx)?;
        self.decoder.forward(&z_q)
    }
}
