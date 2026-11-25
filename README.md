# ncm-rs

Pure Rust port of [HazyResearch/ncm](https://github.com/HazyResearch/ncm) (**Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation**) built on [Candle](https://github.com/huggingface/candle).

`ncm-rs` loads PyTorch `.pt` checkpoints, BPE/Sudoku/HuggingFace tokenizers, and `.pickle` evaluation datasets directly in Rust—with no Python runtime required—and supports CPU, Apple Silicon (`metal`), Apple `accelerate`, and NVIDIA (`cuda`) backends.

## Features

- **Model & Codec (`src/models/`)**:
  - `Vqvae` + `MultiscaleResidualVectorQuantizer` (`Phi` / `PhiConvStack` / `PhiShared` / `PhiPartiallyShared` / `PhiNonShared`, cascade L1 down/upsamplers, precomputed linear/area resize matrices).
  - `NextConceptTransformer` with prefix KV caching, `query_level0` (parallel and `l0_autoregressive`), `predict_level0`, classifier-free guidance (`cfg_scale`), `soft_decisions`, `post_upsample_conv`, `ctx_encoder`, `TiedCodebook` / `Shared` / `Linear` output heads, RoPE, GQA, QK-norm, and level-by-level trajectory decoding.
- **Tokenizers (`src/tokenization/`)**:
  - `SudokuTokenizer`, `BPETokenizer` (loads `.pickle` vocab and merges), and `HFTokenizer` (`tokenizers` + `hf-hub`).
- **CLI Subcommands (`src/main.rs`)**:
  - `generate`: terminal sample generation (with `--trajectory`, `--out` JSONL, CFG, temperature/top-k/top-p).
  - `evaluate`: task scoring for Sudoku (`exact`), XSum (`rouge` + `genPPL`), WMT14 (`bleu` / `chrF`), and unconditional text (`genppl` / `genppl_flm` + entropy via GPT-2 safetensors).
  - `benchmark`: warmup + median wall-clock latency/throughput measurement.
- **Penalized Argmax (`src/generation.rs`)**:
  - Frequency penalty, frequency deficit with allowance, and decaying adjacency window penalty.

## Building

```bash
# CPU
cargo build --release

# Apple Silicon (Metal GPU)
cargo build --release --features metal

# macOS Accelerate
cargo build --release --features accelerate

# NVIDIA CUDA
cargo build --release --features cuda
```

## Usage

### 1. Download checkpoints (and optional datasets)

```bash
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" --local-dir artifacts
```

### 2. Generate

```bash
cargo run --release --features metal -- generate \
  --config configs/generation/tinystories.yaml \
  --checkpoint-dir artifacts/tinystories
```

With trajectory output and JSONL export:

```bash
cargo run --release --features metal -- generate \
  --config configs/generation/sudoku-hard.yaml \
  --checkpoint-dir artifacts/sudoku/hard \
  --data-dir data/sudoku/hard \
  --trajectory \
  --out outputs/sudoku.jsonl
```

### 3. Evaluate

```bash
# Sudoku
cargo run --release --features metal -- evaluate \
  --config config/eval/sudoku-hard.yaml \
  --checkpoint-dir artifacts/sudoku/hard \
  --data-dir data/sudoku/hard

# XSum (ROUGE + GPT-2 genPPL; pass --no-genppl to skip GPT-2 scoring)
cargo run --release --features metal -- evaluate \
  --config config/eval/xsum.yaml \
  --checkpoint-dir artifacts/xsum \
  --data-dir data/xsum

# Unconditional text (TinyStories / OWT / LM1B)
cargo run --release --features metal -- evaluate \
  --config config/eval/tinystories.yaml \
  --checkpoint-dir artifacts/tinystories
```

### 4. Benchmark

```bash
cargo run --release --features metal -- benchmark \
  --config config/eval/tinystories.yaml \
  --checkpoint-dir artifacts/tinystories \
  --batch-size 16 \
  --samples 64 \
  --repeats 3
```

## Testing

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

## License

MIT — see the upstream [HazyResearch/ncm](https://github.com/HazyResearch/ncm) repository for paper, checkpoints, and original MIT license.
