use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use ncm_rs::config::{
    LevelsConfig, QuantResiConfig, RatioSpec, TransformerConfig, VQVAEConfig,
    VectorQuantizerConfig, compute_levels, load_generation_config, load_model_config,
};
use ncm_rs::display::sudoku_board;
use ncm_rs::generation::{TokenPickerConfig, pick_tokens};
use ncm_rs::models::next_concept::{GenerateOptions, NextConceptTransformer};
use ncm_rs::models::quant::{ResizeMode, build_resize_matrix};
use ncm_rs::models::vqvae::Vqvae;
use ncm_rs::tokenization::PreTokenizer;
use ncm_rs::tokenization::sudoku::SudokuTokenizer;

#[test]
fn level_schedules_strictly_increasing() {
    let schedulers = ["linear", "cubic", "exp", "sqrt", "sine", "log", "quadratic"];
    for name in schedulers {
        for n in [2, 3, 5, 8] {
            let levels = compute_levels(name, 1, 64, n).expect("compute_levels");
            assert_eq!(levels.len(), n);
            assert_eq!(levels[0], 1);
            assert_eq!(levels[n - 1], 64);
            for w in levels.windows(2) {
                assert!(w[1] > w[0], "{name} n={n}: {levels:?}");
            }
        }
    }
}

#[test]
fn quadratic_schedule_pinned_values() {
    assert_eq!(
        compute_levels("quadratic", 1, 89, 8).unwrap(),
        vec![1, 2, 8, 17, 29, 45, 65, 89]
    );
    assert_eq!(
        compute_levels("quadratic", 1, 89, 12).unwrap(),
        vec![1, 2, 3, 7, 12, 19, 27, 36, 47, 59, 73, 89]
    );
}

#[test]
fn parse_all_eval_and_generation_configs() {
    let eval_dir = Path::new("config/eval");
    for entry in std::fs::read_dir(eval_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) == Some("yaml") {
            let cfg = load_model_config(&path).unwrap_or_else(|e| {
                panic!("Failed to parse {}: {e}", path.display());
            });
            assert_ne!(cfg.task, "");
        }
    }

    let gen_dir = Path::new("configs/generation");
    for entry in std::fs::read_dir(gen_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) == Some("yaml") {
            let (_, model_cfg, _) = load_generation_config(&path).unwrap_or_else(|e| {
                panic!("Failed to parse {}: {e}", path.display());
            });
            assert_ne!(model_cfg.task, "");
        }
    }
}

#[test]
fn sudoku_tokenizer_and_board_display() {
    let tok = SudokuTokenizer::new();
    let sample = "<BOS>123456789|_________";
    let ids = tok.encode(sample).unwrap();
    assert_eq!(ids[0], SudokuTokenizer::BOS_TOKEN_ID);
    assert_eq!(ids[1], 1);
    assert_eq!(ids[10], SudokuTokenizer::ROW_SEPARATOR_ID);
    assert_eq!(ids[11], SudokuTokenizer::BLANK_TOKEN_ID);
    assert_eq!(tok.decode(&ids, true).unwrap(), "123456789|_________");

    let mut board_ids = vec![11u32];
    for r in 0..9 {
        board_ids.extend(1..=9);
        if r < 8 {
            board_ids.push(10);
        }
    }
    let rendered = sudoku_board(&board_ids);
    assert!(rendered.contains("│ 1 2 3 │ 4 5 6 │ 7 8 9 │"));
    assert!(!rendered.contains("Invalid row separator"));
}

#[test]
fn resize_matrices_row_sums_to_one() {
    let dev = Device::Cpu;
    for (nin, nout) in [(1, 16), (4, 16), (16, 4), (13, 37)] {
        for mode in [ResizeMode::Linear, ResizeMode::Area] {
            let m = build_resize_matrix(nin, nout, mode, &dev).unwrap();
            let row_sums = m.sum(1).unwrap().to_vec1::<f32>().unwrap();
            for s in row_sums {
                assert!((s - 1.0).abs() < 1e-5, "nin={nin} nout={nout} sum={s}");
            }
        }
    }
}

#[test]
fn penalized_argmax_frequency_and_adjacency() {
    let dev = Device::Cpu;
    // 1 batch, 3 positions, 3 vocab items. Token 0 always has highest raw logit (10.0).
    let logits = Tensor::from_vec(
        vec![10.0f32, 9.0, 8.0, 10.0, 9.0, 8.0, 10.0, 9.0, 8.0],
        (1, 3, 3),
        &dev,
    )
    .unwrap();

    // Without penalties, picks [0, 0, 0].
    let plain = pick_tokens(
        &logits,
        &TokenPickerConfig::from_settings(0.0, 0.0, None, None),
    )
    .unwrap()
    .to_vec2::<u32>()
    .unwrap();
    assert_eq!(plain, vec![vec![0, 0, 0]]);

    // With frequency penalty 5.0, picks [0, 1, 2].
    let freq = pick_tokens(
        &logits,
        &TokenPickerConfig::from_settings(5.0, 0.0, None, None),
    )
    .unwrap()
    .to_vec2::<u32>()
    .unwrap();
    assert_eq!(freq, vec![vec![0, 1, 2]]);

    // With adjacency penalty 5.0, window 1: [0, 1, 0].
    let adj = pick_tokens(
        &logits,
        &TokenPickerConfig::from_settings(0.0, 0.0, Some("5.0,1,1.0"), None),
    )
    .unwrap()
    .to_vec2::<u32>()
    .unwrap();
    assert_eq!(adj, vec![vec![0, 1, 0]]);
}

#[test]
fn end_to_end_synthetic_ncm_generate() {
    let dev = Device::Cpu;
    let embed_dim = 16;
    let n_embd = 32;
    let in_vocab = 12;
    let vq_vocab = 8;
    let levels = vec![2, 4, 8];
    let prefix_len = 4;

    let vqvae_cfg = VQVAEConfig {
        n_layers: 2,
        embed_dim,
        in_vocab_size: in_vocab,
        context_length: 8,
        compression_factor: 1,
        dropout: 0.0,
        pre_quant_groupnorm: Some(4),
        pre_quant_norm_type: "group".to_string(),
        pre_quant_dropout: None,
        use_qk_norm: true,
        enc_layers: 0,
        enc_attn_window: 0,
        enc_conv_layers: 0,
        enc_conv_kernel: 3,
        dec_layers: 1,
        enc_positional: true,
        dec_positional: false,
        vqvae_vocab_size: vec![vq_vocab, vq_vocab, vq_vocab],
        checkpoint_path: None,
        vector_quantizer_config: VectorQuantizerConfig {
            clss: "multiscale_residual_vector_quantizer".to_string(),
            decay: 0.99,
            epsilon: 1e-5,
            commitment_cost: 0.25,
            learned_l1_sampling: true,
            learned_all_sampling: false,
            learned_l1_cascade: false,
            learned_l1_cascade_base: 4,
            learned_l1_cascade_norm: false,
            learned_l1_cascade_norm_final: false,
            quant_resi: Some(QuantResiConfig {
                enabled: true,
                ratio: RatioSpec::Scalar(0.5),
                share_mode: 0,
                learnable_ratio: false,
                arch: "blend".to_string(),
                hidden_mult: 1,
                kernel_size: 3,
            }),
            levels: LevelsConfig {
                use_manual_levels: true,
                manual_levels: levels.clone(),
                scheduler: "linear".to_string(),
                start: 2,
                end: 8,
                num_levels: 3,
            },
        },
    };

    let model_cfg = TransformerConfig {
        n_layer: 2,
        n_head: 2,
        n_kv_heads: None,
        bias: true,
        dropout: 0.0,
        n_embd,
        context_length: prefix_len + 14,
        prefix_len,
        raw_token_prefix: true,
        contextual_prefix: false,
        prefix_pad_token_id: Some(0),
        vocab_size: vec![vq_vocab, vq_vocab, vq_vocab],
        predict_level0: false,
        attn_pattern: "block_diagonal".to_string(),
        use_positional_encoding: true,
        use_level_encoding: true,
        use_rope: true,
        rope_base: 10_000,
        use_qk_norm: true,
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
        query_level0: true,
        l0_autoregressive: false,
        shared_output_head: false,
        shared_head_per_level_bias: false,
        untied_head_bias: true,
        tie_heads_to_codebook: true,
        tie_heads_learn_scale: true,
        tie_heads_trainable: false,
        tie_heads_bias: false,
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let vqvae = Arc::new(Vqvae::load(&vqvae_cfg, vb.pp("vqvae")).unwrap());
    let model = NextConceptTransformer::load(Arc::clone(&vqvae), model_cfg, vb.pp("ncm")).unwrap();

    let prefix = Tensor::from_vec(vec![1u32, 2, 3, 0, 4, 5, 0, 0], (2, prefix_len), &dev).unwrap();
    let opts = GenerateOptions {
        temperature: 0.8,
        top_k: 4,
        top_p: 0.9,
        use_kv_cache: true,
        cfg_scale: 1.5,
        cfg_null_eos: 11,
        soft_decisions: true,
        soft_temperature: 1.0,
        return_trajectory: true,
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let (codes, traj) = model.generate(None, &prefix, &opts, &mut rng).unwrap();
    assert_eq!(codes.dims2().unwrap(), (2, 14));
    assert_eq!(traj.dims4().unwrap(), (2, 3, 8, embed_dim));

    let final_lat = traj.narrow(1, 2, 1).unwrap().squeeze(1).unwrap();
    let dec_logits = vqvae.decoder.forward(&final_lat).unwrap();
    assert_eq!(dec_logits.dims3().unwrap(), (2, 8, in_vocab));
}

#[test]
fn bpe_tokenizer_encode_decode_roundtrip() {
    use ncm_rs::tokenization::bpe::BPETokenizer;
    use std::collections::HashMap;

    let mut vocab = HashMap::new();
    vocab.insert(0u32, b"<|endoftext|>".to_vec());
    vocab.insert(1u32, b"<|pad|>".to_vec());
    vocab.insert(2u32, b"h".to_vec());
    vocab.insert(3u32, b"i".to_vec());
    vocab.insert(4u32, b"!".to_vec());
    vocab.insert(5u32, b"hi".to_vec());

    let merges = vec![(b"h".to_vec(), b"i".to_vec())];
    let mut special = HashMap::new();
    special.insert("eos_token".to_string(), "<|endoftext|>".to_string());
    special.insert("pad_token".to_string(), "<|pad|>".to_string());

    let tok = BPETokenizer::new(vocab, merges, &special).unwrap();
    assert_eq!(tok.eos_token_id(), Some(0));
    assert_eq!(tok.pad_token_id(), Some(1));

    let ids = tok.encode("hi!<|endoftext|>").unwrap();
    assert_eq!(ids, vec![5, 4, 0]);
    assert_eq!(tok.decode(&ids, true).unwrap(), "hi!");
    assert_eq!(tok.decode(&ids, false).unwrap(), "hi!<|endoftext|>");
}

#[test]
fn penalized_argmax_allowance_and_deficit() {
    let dev = Device::Cpu;
    // 1 batch, 3 positions, 2 vocab items. Token 0 always has slightly higher raw logit (1.0 vs 0.8).
    let logits = Tensor::from_vec(vec![1.0f32, 0.8, 1.0, 0.8, 1.0, 0.8], (1, 3, 2), &dev).unwrap();
    // Allowance: token 0 gets 1/3 (1.0 over length 3), token 1 gets 2/3 (2.0 over length 3).
    // With deficit=1.0 and frequency=2.0, after picking token 0 once, token 1 gets deficit bonus and token 0 gets excess penalty.
    let cfg = TokenPickerConfig::from_settings(2.0, 1.0, None, Some(vec![1.0 / 3.0, 2.0 / 3.0]));
    let picked = pick_tokens(&logits, &cfg)
        .unwrap()
        .to_vec2::<u32>()
        .unwrap();
    assert_eq!(picked, vec![vec![1, 0, 1]]);
}

#[test]
fn vqvae_encode_decode_with_compression_and_conv_blocks() {
    let dev = Device::Cpu;
    let vqvae_cfg = VQVAEConfig {
        n_layers: 2,
        embed_dim: 16,
        in_vocab_size: 32,
        context_length: 16,
        compression_factor: 2,
        dropout: 0.0,
        pre_quant_groupnorm: Some(4),
        pre_quant_norm_type: "layer".to_string(),
        pre_quant_dropout: None,
        use_qk_norm: true,
        enc_layers: 1,
        enc_attn_window: 3,
        enc_conv_layers: 1,
        enc_conv_kernel: 3,
        dec_layers: 2,
        enc_positional: true,
        dec_positional: true,
        vqvae_vocab_size: vec![8, 8, 8],
        checkpoint_path: None,
        vector_quantizer_config: VectorQuantizerConfig {
            clss: "multiscale_residual_vector_quantizer".to_string(),
            decay: 0.99,
            epsilon: 1e-5,
            commitment_cost: 0.25,
            learned_l1_sampling: true,
            learned_all_sampling: false,
            learned_l1_cascade: true,
            learned_l1_cascade_base: 2,
            learned_l1_cascade_norm: true,
            learned_l1_cascade_norm_final: true,
            quant_resi: Some(QuantResiConfig {
                enabled: true,
                ratio: RatioSpec::Scalar(0.5),
                share_mode: 1,
                learnable_ratio: true,
                arch: "blend".to_string(),
                hidden_mult: 1,
                kernel_size: 3,
            }),
            levels: LevelsConfig {
                use_manual_levels: true,
                manual_levels: vec![2, 4, 8],
                scheduler: "linear".to_string(),
                start: 2,
                end: 8,
                num_levels: 3,
            },
        },
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let vqvae = Vqvae::load(&vqvae_cfg, vb.pp("vqvae")).unwrap();

    let x = Tensor::from_vec(
        (0..32u32).map(|i| i % 32).collect::<Vec<_>>(),
        (2, 16),
        &dev,
    )
    .unwrap();
    let idx = vqvae.encode(&x).unwrap();
    assert_eq!(idx.dims2().unwrap(), (2, 14));
    let logits = vqvae.decode(&idx).unwrap();
    assert_eq!(logits.dims3().unwrap(), (2, 16, 32));
}

#[test]
fn ncm_predict_level0_ctx_encoder_and_l0_ar() {
    use ncm_rs::config::{CtxEncoderConfig, PostUpsampleConvConfig};

    let dev = Device::Cpu;
    let embed_dim = 16;
    let n_embd = 32;
    let in_vocab = 16;
    let vq_vocab = 8;
    let prefix_len = 4;

    let vqvae_cfg = VQVAEConfig {
        n_layers: 2,
        embed_dim,
        in_vocab_size: in_vocab,
        context_length: 8,
        compression_factor: 1,
        dropout: 0.0,
        pre_quant_groupnorm: None,
        pre_quant_norm_type: "group".to_string(),
        pre_quant_dropout: None,
        use_qk_norm: false,
        enc_layers: 0,
        enc_attn_window: 0,
        enc_conv_layers: 0,
        enc_conv_kernel: 3,
        dec_layers: 1,
        enc_positional: false,
        dec_positional: false,
        vqvae_vocab_size: vec![vq_vocab, vq_vocab, vq_vocab],
        checkpoint_path: None,
        vector_quantizer_config: VectorQuantizerConfig {
            clss: "multiscale_residual_vector_quantizer".to_string(),
            decay: 0.99,
            epsilon: 1e-5,
            commitment_cost: 0.25,
            learned_l1_sampling: false,
            learned_all_sampling: false,
            learned_l1_cascade: false,
            learned_l1_cascade_base: 4,
            learned_l1_cascade_norm: false,
            learned_l1_cascade_norm_final: false,
            quant_resi: Some(QuantResiConfig {
                enabled: true,
                ratio: RatioSpec::Scalar(0.5),
                share_mode: 0,
                learnable_ratio: false,
                arch: "convstack".to_string(),
                hidden_mult: 1,
                kernel_size: 3,
            }),
            levels: LevelsConfig {
                use_manual_levels: true,
                manual_levels: vec![2, 4, 8],
                scheduler: "linear".to_string(),
                start: 2,
                end: 8,
                num_levels: 3,
            },
        },
    };

    // Variant A: predict_level0 + ctx_encoder + shared_output_head + learned positional encoding
    let cfg_a = TransformerConfig {
        n_layer: 1,
        n_head: 2,
        n_kv_heads: Some(1),
        bias: true,
        dropout: 0.0,
        n_embd,
        context_length: prefix_len + 12,
        prefix_len,
        raw_token_prefix: false,
        contextual_prefix: false,
        prefix_pad_token_id: Some(0),
        vocab_size: vec![vq_vocab, vq_vocab],
        predict_level0: true,
        attn_pattern: "block_causal".to_string(),
        use_positional_encoding: true,
        use_level_encoding: true,
        use_rope: false,
        rope_base: 10_000,
        use_qk_norm: true,
        use_swiglu: true,
        use_geglu: false,
        use_relu2: false,
        mlp_ratio: 4.0,
        parallel_block: true,
        normformer: true,
        use_layerscale: true,
        layerscale_init: 1e-4,
        use_rezero: false,
        ctx_encoder: Some(CtxEncoderConfig {
            enabled: true,
            n_layer: 1,
            vocab_size: Some(in_vocab),
            pretrained: None,
            finetune: false,
            pad_id: 0,
        }),
        post_upsample_conv: Some(PostUpsampleConvConfig {
            enabled: true,
            kernel_size: 3,
        }),
        use_flex_attention: false,
        query_level0: false,
        l0_autoregressive: false,
        shared_output_head: true,
        shared_head_per_level_bias: true,
        untied_head_bias: true,
        tie_heads_to_codebook: false,
        tie_heads_learn_scale: false,
        tie_heads_trainable: false,
        tie_heads_bias: false,
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let vqvae = Arc::new(Vqvae::load(&vqvae_cfg, vb.pp("vqvae")).unwrap());
    let model_a = NextConceptTransformer::load(Arc::clone(&vqvae), cfg_a, vb.pp("a")).unwrap();

    let prefix = Tensor::from_vec(vec![1u32, 2, 3, 0], (1, prefix_len), &dev).unwrap();
    let opts = GenerateOptions {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        use_kv_cache: false,
        cfg_scale: 0.0,
        cfg_null_eos: 0,
        soft_decisions: false,
        soft_temperature: 1.0,
        return_trajectory: false,
    };
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let (codes_a, lat_a) = model_a.generate(None, &prefix, &opts, &mut rng).unwrap();
    assert_eq!(codes_a.dims2().unwrap(), (1, 14));
    assert_eq!(lat_a.dims3().unwrap(), (1, 8, embed_dim));

    // Variant B: query_level0 + l0_autoregressive + kv_cache
    let cfg_b = TransformerConfig {
        n_layer: 1,
        n_head: 2,
        n_kv_heads: None,
        bias: false,
        dropout: 0.0,
        n_embd,
        context_length: prefix_len + 14,
        prefix_len,
        raw_token_prefix: true,
        contextual_prefix: false,
        prefix_pad_token_id: None,
        vocab_size: vec![vq_vocab, vq_vocab, vq_vocab],
        predict_level0: false,
        attn_pattern: "block_diagonal".to_string(),
        use_positional_encoding: true,
        use_level_encoding: false,
        use_rope: true,
        rope_base: 10_000,
        use_qk_norm: false,
        use_swiglu: false,
        use_geglu: true,
        use_relu2: false,
        mlp_ratio: 4.0,
        parallel_block: false,
        normformer: false,
        use_layerscale: false,
        layerscale_init: 1e-4,
        use_rezero: true,
        ctx_encoder: None,
        post_upsample_conv: None,
        use_flex_attention: false,
        query_level0: true,
        l0_autoregressive: true,
        shared_output_head: false,
        shared_head_per_level_bias: false,
        untied_head_bias: false,
        tie_heads_to_codebook: false,
        tie_heads_learn_scale: false,
        tie_heads_trainable: false,
        tie_heads_bias: false,
    };
    let model_b = NextConceptTransformer::load(vqvae, cfg_b, vb.pp("b")).unwrap();
    let opts_kv = GenerateOptions {
        use_kv_cache: true,
        ..opts
    };
    let (codes_b, lat_b) = model_b.generate(None, &prefix, &opts_kv, &mut rng).unwrap();
    assert_eq!(codes_b.dims2().unwrap(), (1, 14));
    assert_eq!(lat_b.dims3().unwrap(), (1, 8, embed_dim));
}

#[test]
fn format_sample_with_trajectory() {
    use ncm_rs::display::format_sample;
    use ncm_rs::generation::{SampleRecord, TrajectoryStep};

    let rec = SampleRecord {
        task: "tinystories".to_string(),
        seed: 7,
        batch_size: 1,
        precision: "fp32".to_string(),
        dataset_index: None,
        prompt_ids: None,
        prompt: None,
        prompt_truncated: None,
        reference: None,
        reference_ids: None,
        output: "Once upon a time.".to_string(),
        raw_ids: vec![1, 2, 3],
        trajectory: Some(vec![
            TrajectoryStep {
                level: 1,
                codes: 1,
                text: "Once".to_string(),
                raw_ids: vec![1],
            },
            TrajectoryStep {
                level: 2,
                codes: 3,
                text: "Once upon a time.".to_string(),
                raw_ids: vec![1, 2, 3],
            },
        ]),
    };
    let formatted = format_sample(&rec, 1, 80);
    assert!(formatted.contains("Sample 1 · tinystories · seed 7"));
    assert!(formatted.contains("Level 1/2 · 1 codes"));
    assert!(formatted.contains("Once upon a time."));
}

#[test]
fn evaluation_metrics_exact_rouge_bleu_and_entropy() {
    use ncm_rs::evaluation::{evaluate_bleu, evaluate_exact, evaluate_rouge, token_entropy};

    // Exact Sudoku
    let valid_board =
        "534678912|672195348|198342567|859761423|426853791|713924856|961537284|287419635|345286179";
    let exact = evaluate_exact(&[valid_board.to_string()], &[valid_board.to_string()]);
    assert_eq!(exact.solved, 1);
    assert!((exact.solved_pct - 100.0).abs() < 1e-6);
    assert!((exact.valid_pct - 100.0).abs() < 1e-6);

    // ROUGE
    let rouge = evaluate_rouge(
        &["the cat sat on the mat".to_string()],
        &["the cat is on the mat".to_string()],
    );
    assert!(rouge.rouge1 > 80.0);
    assert!(rouge.rouge_l > 80.0);

    // BLEU
    let bleu = evaluate_bleu(
        &["The quick brown fox jumps over the lazy dog.".to_string()],
        &["The quick brown fox jumps over the lazy dog.".to_string()],
    );
    assert!((bleu.bleu_cased - 100.0).abs() < 1e-6);
    assert!((bleu.chrf - 100.0).abs() < 1e-6);

    // Entropy: uniform over 2 tokens = ln(2)
    let ent = token_entropy(&[1, 2, 1, 2]);
    assert!((ent - std::f64::consts::LN_2).abs() < 1e-6);
}
