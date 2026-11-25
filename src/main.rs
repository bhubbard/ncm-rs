use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};
use candle_core::{DType, Device};
use clap::{Parser, Subcommand, ValueEnum};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use ncm_rs::config::{GenerationFileConfig, load_generation_config, load_model_config};
use ncm_rs::display::format_sample;
use ncm_rs::evaluation::{
    Gpt2Scorer, evaluate_bleu, evaluate_exact, evaluate_rouge, token_entropy,
};
use ncm_rs::generation::{Generator, load_inputs};

#[derive(Parser, Debug)]
#[command(name = "ncm-rs", about = "Next-Concept Modeling (NCM) in Pure Rust")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Generate readable examples without running evaluation metrics.
    Generate(GenerateArgs),
    /// Evaluate a checkpoint on a task dataset and print task-specific metrics.
    Evaluate(EvaluateArgs),
    /// Benchmark generation wall-clock latency and throughput.
    Benchmark(BenchmarkArgs),
}

#[derive(ValueEnum, Debug, Clone, Copy)]
enum PrecisionArg {
    Fp32,
    Bf16,
    Fp16,
}

#[derive(ValueEnum, Debug, Clone, Copy)]
enum DeviceArg {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

#[derive(Parser, Debug)]
struct GenerateArgs {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    checkpoint_dir: Option<PathBuf>,
    #[arg(long)]
    data_dir: Option<PathBuf>,
    #[arg(long)]
    samples: Option<usize>,
    #[arg(long)]
    batch_size: Option<usize>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    trajectory: bool,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    temperature: Option<f64>,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long)]
    top_p: Option<f64>,
    #[arg(long)]
    cfg: Option<f64>,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    kv_cache: bool,
    #[arg(long, value_enum, default_value_t = PrecisionArg::Fp32)]
    precision: PrecisionArg,
    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,
}

#[derive(Parser, Debug)]
struct EvaluateArgs {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    checkpoint_dir: PathBuf,
    #[arg(long)]
    data_dir: Option<PathBuf>,
    #[arg(long)]
    n: Option<usize>,
    #[arg(long)]
    bs: Option<usize>,
    #[arg(long)]
    cfg: Option<f64>,
    #[arg(long, default_value_t = false)]
    no_genppl: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    kv_cache: bool,
    #[arg(long, value_enum, default_value_t = PrecisionArg::Fp32)]
    precision: PrecisionArg,
    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,
}

#[derive(Parser, Debug)]
struct BenchmarkArgs {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    checkpoint_dir: PathBuf,
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
    #[arg(long, default_value_t = 64)]
    samples: usize,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    kv_cache: bool,
    #[arg(long, value_enum, default_value_t = PrecisionArg::Fp32)]
    precision: PrecisionArg,
    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,
}

fn select_device(arg: DeviceArg) -> Result<Device> {
    match arg {
        DeviceArg::Cpu => Ok(Device::Cpu),
        DeviceArg::Metal => Ok(Device::new_metal(0)?),
        DeviceArg::Cuda => Ok(Device::new_cuda(0)?),
        DeviceArg::Auto => {
            if let Ok(d) = Device::new_cuda(0) {
                Ok(d)
            } else if let Ok(d) = Device::new_metal(0) {
                Ok(d)
            } else {
                Ok(Device::Cpu)
            }
        }
    }
}

fn select_dtype(arg: PrecisionArg) -> (DType, &'static str) {
    match arg {
        PrecisionArg::Fp32 => (DType::F32, "fp32"),
        PrecisionArg::Bf16 => (DType::BF16, "bf16"),
        PrecisionArg::Fp16 => (DType::F16, "fp16"),
    }
}

fn run_generate(args: GenerateArgs) -> Result<()> {
    let (mut gen_cfg, mut model_cfg, _) = load_generation_config(&args.config)?;
    if let Some(s) = args.samples {
        gen_cfg.samples = s;
    }
    if let Some(bs) = args.batch_size {
        gen_cfg.batch_size = bs;
    }
    if let Some(seed) = args.seed {
        gen_cfg.seed = seed;
    }
    if let Some(t) = args.temperature {
        model_cfg.generation.temperature = t;
    }
    if let Some(k) = args.top_k {
        model_cfg.generation.top_k = k;
    }
    if let Some(p) = args.top_p {
        model_cfg.generation.top_p = p;
    }
    if let Some(c) = args.cfg {
        model_cfg.generation.cfg_scale = c;
    }

    let ckpt_dir = args
        .checkpoint_dir
        .or_else(|| gen_cfg.checkpoint_dir.as_deref().map(PathBuf::from))
        .ok_or_else(|| anyhow::anyhow!("Missing checkpoint_dir"))?;
    let data_dir = args
        .data_dir
        .or_else(|| gen_cfg.data_dir.as_deref().map(PathBuf::from));

    if let Some(ref out) = args.out
        && out.exists()
    {
        bail!("Output already exists: {}", out.display());
    }

    let device = select_device(args.device)?;
    let (dtype, prec_label) = select_dtype(args.precision);
    eprintln!(
        "Loading {}; preparing {} example(s) on {device:?} ({prec_label})…",
        model_cfg.task, gen_cfg.samples
    );

    let mut rng = ChaCha8Rng::seed_from_u64(gen_cfg.seed);
    let inputs = match data_dir {
        Some(ref dir) => load_inputs(dir, &model_cfg, &gen_cfg.split, gen_cfg.samples, &mut rng)?,
        None => None,
    };

    let generator = Generator::load(model_cfg, &ckpt_dir, device, dtype, prec_label)?;
    let records = generator.generate_samples(
        &gen_cfg,
        inputs.as_deref(),
        args.trajectory,
        args.kv_cache,
        &mut rng,
    )?;

    let mut out_file = if let Some(ref out_path) = args.out {
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Some(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(out_path)?,
        )
    } else {
        None
    };

    for (idx, rec) in records.iter().enumerate() {
        println!("{}\n", format_sample(rec, idx + 1, 96));
        if let Some(ref mut f) = out_file {
            writeln!(f, "{}", serde_json::to_string(rec)?)?;
        }
    }
    Ok(())
}

fn run_evaluate(args: EvaluateArgs) -> Result<()> {
    let mut model_cfg = load_model_config(&args.config)?;
    if let Some(c) = args.cfg {
        model_cfg.generation.cfg_scale = c;
    }
    let device = select_device(args.device)?;
    let (dtype, prec_label) = select_dtype(args.precision);
    let generator = Generator::load(
        model_cfg.clone(),
        &args.checkpoint_dir,
        device.clone(),
        dtype,
        prec_label,
    )?;

    let split = model_cfg
        .eval
        .split
        .clone()
        .unwrap_or_else(|| "test".to_string());
    let bs = args.bs.or(model_cfg.eval.batch_size).unwrap_or(16);
    let metric = model_cfg
        .eval
        .metric
        .clone()
        .unwrap_or_else(|| "exact".to_string());

    if let Some(ref data_dir) = args.data_dir {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let max_n = args.n.or(model_cfg.eval.n_docs).unwrap_or(usize::MAX);
        let inputs = load_inputs(data_dir, &model_cfg, &split, max_n, &mut rng)?
            .ok_or_else(|| anyhow::anyhow!("No data configured for split {split}"))?;
        let gen_cfg = GenerationFileConfig {
            model_config: args.config.display().to_string(),
            checkpoint_dir: Some(args.checkpoint_dir.display().to_string()),
            data_dir: Some(data_dir.display().to_string()),
            split,
            samples: inputs.len(),
            batch_size: bs,
            seed: 0,
            generation: None,
        };
        let records =
            generator.generate_samples(&gen_cfg, Some(&inputs), false, args.kv_cache, &mut rng)?;
        let hyps: Vec<String> = records.iter().map(|r| r.output.clone()).collect();
        let refs: Vec<String> = records
            .iter()
            .map(|r| r.reference.clone().unwrap_or_default())
            .collect();

        match metric.as_str() {
            "exact" => {
                let m = evaluate_exact(&hyps, &refs);
                println!(
                    "{:>9} {:>7} {:>7} {:>8}",
                    "solved %", "solved", "cell %", "valid %"
                );
                println!(
                    "{:9.2} {:4}/{} {:7.2} {:8.2}",
                    m.solved_pct, m.solved, m.total, m.cell_pct, m.valid_pct
                );
            }
            "rouge" => {
                let m = evaluate_rouge(&hyps, &refs);
                let gp = if args.no_genppl {
                    f64::NAN
                } else {
                    let ref_id = model_cfg.eval.ref_model.as_deref().unwrap_or("gpt2");
                    Gpt2Scorer::load(ref_id, device, dtype)?.mean_ppl(&hyps, None)?
                };
                println!(
                    "{:>7} {:>7} {:>7} {:>8} {:>7}",
                    "R-1", "R-2", "R-L", "genPPL", "words"
                );
                println!(
                    "{:7.2} {:7.2} {:7.2} {:8.2} {:7.1}",
                    m.rouge1, m.rouge2, m.rouge_l, gp, m.mean_words
                );
            }
            "bleu" => {
                let m = evaluate_bleu(&hyps, &refs);
                println!(
                    "{:>8} {:>9} {:>7} {:>10}",
                    "BLEU", "BLEU(lc)", "chrF", "len ratio"
                );
                println!(
                    "{:8.2} {:9.2} {:7.2} {:10.3}",
                    m.bleu_cased, m.bleu_lowercase, m.chrf, m.len_ratio
                );
            }
            other => bail!("Unsupported conditional metric: {other}"),
        }
    } else {
        let seeds = if model_cfg.eval.seeds.is_empty() {
            vec![0]
        } else {
            model_cfg.eval.seeds.clone()
        };
        let n_samples = args.n.or(model_cfg.eval.n_samples).unwrap_or(64);
        let scorer = if args.no_genppl {
            None
        } else {
            let ref_id = model_cfg.eval.ref_model.as_deref().unwrap_or("gpt2");
            Some(Gpt2Scorer::load(ref_id, device, dtype)?)
        };
        let max_len = if metric == "genppl_flm" {
            Some(128)
        } else {
            None
        };
        println!("{:>5} {:>8} {:>8}", "seed", "genPPL", "entropy");
        let mut ppls = Vec::new();
        let mut ents = Vec::new();
        for seed in seeds {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let gen_cfg = GenerationFileConfig {
                model_config: args.config.display().to_string(),
                checkpoint_dir: Some(args.checkpoint_dir.display().to_string()),
                data_dir: None,
                split: split.clone(),
                samples: n_samples,
                batch_size: bs,
                seed,
                generation: None,
            };
            let records =
                generator.generate_samples(&gen_cfg, None, false, args.kv_cache, &mut rng)?;
            let texts: Vec<String> = if metric == "genppl_flm" {
                records
                    .iter()
                    .map(|r| generator.pre_tokenizer.decode(&r.raw_ids, false))
                    .collect::<Result<Vec<_>>>()?
            } else {
                records.iter().map(|r| r.output.clone()).collect()
            };
            let ent: f64 = records
                .iter()
                .map(|r| token_entropy(&r.raw_ids))
                .sum::<f64>()
                / records.len().max(1) as f64;
            let ppl = match scorer {
                Some(ref s) => s.mean_ppl(&texts, max_len)?,
                None => f64::NAN,
            };
            println!("{seed:5} {ppl:8.2} {ent:8.4}");
            ppls.push(ppl);
            ents.push(ent);
        }
        let mean_ppl = ppls.iter().sum::<f64>() / ppls.len().max(1) as f64;
        let mean_ent = ents.iter().sum::<f64>() / ents.len().max(1) as f64;
        println!("\n{:>5} {mean_ppl:8.2} {mean_ent:8.4}", "mean");
    }
    Ok(())
}

fn run_benchmark(args: BenchmarkArgs) -> Result<()> {
    let model_cfg = load_model_config(&args.config)?;
    let device = select_device(args.device)?;
    let (dtype, prec_label) = select_dtype(args.precision);
    let generator = Generator::load(
        model_cfg.clone(),
        &args.checkpoint_dir,
        device,
        dtype,
        prec_label,
    )?;

    let gen_cfg = ncm_rs::config::GenerationFileConfig {
        model_config: args.config.display().to_string(),
        checkpoint_dir: Some(args.checkpoint_dir.display().to_string()),
        data_dir: None,
        split: "test".to_string(),
        samples: args.samples,
        batch_size: args.batch_size,
        seed: args.seed,
        generation: None,
    };

    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    for _ in 0..args.warmup {
        let mut warm_cfg = gen_cfg.clone();
        warm_cfg.samples = args.batch_size;
        let _ = generator.generate_samples(&warm_cfg, None, false, args.kv_cache, &mut rng)?;
    }

    let mut times_ms = Vec::with_capacity(args.repeats);
    for _ in 0..args.repeats {
        let start = Instant::now();
        let _ = generator.generate_samples(&gen_cfg, None, false, args.kv_cache, &mut rng)?;
        times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    times_ms.sort_by(f64::total_cmp);
    let median = times_ms[times_ms.len() / 2];
    println!(
        "Task: {} | B={} N={} | median: {:.2} ms ({:.3} s)",
        model_cfg.task,
        args.batch_size,
        args.samples,
        median,
        median / 1000.0
    );
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Generate(args) => run_generate(args),
        Command::Evaluate(args) => run_evaluate(args),
        Command::Benchmark(args) => run_benchmark(args),
    }
}
