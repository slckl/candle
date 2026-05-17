#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::gemma4::{
    config::{Gemma4Config, Gemma4TextConfig},
    text::TextModel,
    Model,
};

use candle::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[allow(clippy::large_enum_variant)]
enum ModelKind {
    TextOnly(TextModel),
    Multimodal(Model),
}

struct TextGeneration {
    model: ModelKind,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: ModelKind,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = {
            let temperature = temp.unwrap_or(0.);
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (top_k, top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(seed, sampling)
        };

        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(
        &mut self,
        prompt: &str,
        sample_len: usize,
        pixel_values: Option<Vec<Tensor>>,
    ) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<eos>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <eos> token"),
        };
        let end_of_turn_token = self.tokenizer.get_token("<turn|>");
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            // Vision/audio features are only injected on the prefill step
            // (`index == 0`); subsequent decode steps go through the plain
            // text path so the model just keeps generating from KV cache.
            let logits = match &mut self.model {
                ModelKind::TextOnly(m) => m.forward(&input, start_pos)?,
                ModelKind::Multimodal(m) => {
                    if index == 0 && pixel_values.is_some() {
                        m.forward_multimodal(
                            &input,
                            pixel_values.as_deref(),
                            None,
                            None,
                            start_pos,
                        )?
                    } else {
                        m.forward(&input, start_pos)?
                    }
                }
            };
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || Some(next_token) == end_of_turn_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    use_flash_attn: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    /// Load the multimodal model (vision + audio encoders).
    #[arg(long)]
    multimodal: bool,

    /// Image file(s) to feed into the model. Each path is loaded, resized
    /// preserving aspect ratio under `--max-soft-tokens`, and inserted ahead
    /// of the user message in the prompt. Implies `--multimodal`. Repeat the
    /// flag to pass several images.
    #[arg(long = "image", value_name = "PATH")]
    images: Vec<String>,

    /// Maximum number of soft image tokens per image. The image is resized
    /// (aspect-ratio preserving) so the resulting patch grid produces at most
    /// this many soft tokens. Matches the gemma-4 image processor's
    /// `max_soft_tokens` (default 256, model cap 280).
    #[arg(long, default_value_t = 256)]
    max_soft_tokens: usize,

    /// Force a square resize to this edge length (in pixels, a multiple of 48)
    /// instead of the aspect-ratio-preserving resize. Useful for debugging.
    #[arg(long)]
    square_image_size: Option<usize>,

    /// Pass the prompt through verbatim instead of wrapping it in the gemma4
    /// instruction-tuned chat template.
    #[arg(long)]
    raw_prompt: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Use the slower dmmv cuda kernel.
    #[arg(long)]
    force_dmmv: bool,
}

/// Aspect-ratio-preserving target size for the gemma-4 image processor.
/// Returns `(height, width)` such that `(height / patch_size) * (width / patch_size)
/// / pooling_kernel_size^2 <= max_soft_tokens` and both dims are multiples of
/// `patch_size * pooling_kernel_size = 48`.
fn aspect_preserving_size(height: usize, width: usize, max_soft_tokens: usize) -> (usize, usize) {
    const PATCH_SIZE: f64 = 16.0;
    const POOLING: f64 = 3.0;
    const SIDE_MULT: f64 = PATCH_SIZE * POOLING; // 48

    let target_patches = (max_soft_tokens as f64) * POOLING * POOLING;
    let target_px = target_patches * PATCH_SIZE * PATCH_SIZE;
    let total_px = (height * width) as f64;
    let factor = (target_px / total_px).sqrt();

    let mut th = ((height as f64 * factor) / SIDE_MULT).floor() as usize * SIDE_MULT as usize;
    let mut tw = ((width as f64 * factor) / SIDE_MULT).floor() as usize * SIDE_MULT as usize;
    // Ensure at least one patch in each dim so the model has something to look at.
    if th == 0 {
        th = SIDE_MULT as usize;
    }
    if tw == 0 {
        tw = SIDE_MULT as usize;
    }
    (th, tw)
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    #[cfg(feature = "cuda")]
    candle::quantized::cuda::set_force_dmmv(args.force_dmmv);

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = args
        .model_id
        .clone()
        .unwrap_or_else(|| "google/gemma-4-E4B-it".to_string());
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
            match candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json") {
                Ok(files) => files,
                Err(_) => vec![repo.get("model.safetensors")?],
            }
        }
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

    // `--image` implies multimodal: we need the vision tower.
    let multimodal = args.multimodal || !args.images.is_empty();
    let model = if multimodal {
        let mut config: Gemma4Config = match args.config_file {
            Some(config_file) => serde_json::from_slice(&std::fs::read(config_file)?)?,
            None => {
                let config_file = repo.get("config.json")?;
                serde_json::from_slice(&std::fs::read(config_file)?)?
            }
        };
        // The audio tower path in this crate doesn't yet handle the
        // `Gemma4ClippableLinear` weight layout used in the released
        // checkpoints, so skip loading it. Vision-only / text-only is enough
        // for now and saves a substantial chunk of memory.
        config.audio_config = None;
        let model = Model::new(&config, vb)?;
        ModelKind::Multimodal(model)
    } else {
        let mut config: Gemma4TextConfig = match args.config_file {
            Some(config_file) => serde_json::from_slice(&std::fs::read(config_file)?)?,
            None => {
                let config_file = repo.get("config.json")?;
                // For text-only, try to parse the text_config sub-object
                let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(config_file)?)?;
                if let Some(text_cfg) = raw.get("text_config") {
                    serde_json::from_value(text_cfg.clone())?
                } else {
                    serde_json::from_value(raw)?
                }
            }
        };
        config.use_flash_attn = args.use_flash_attn;
        // The released gemma-4 checkpoints are multimodal; the language model
        // weights live under `model.language_model.*`.
        let model = TextModel::new(&config, vb.pp("model").pp("language_model"))?;
        ModelKind::TextOnly(model)
    };

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.top_k,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    // Constants from the gemma-4 vision config / image processor.
    const PATCH_SIZE: usize = 16;
    const POOLING_KERNEL_SIZE: usize = 3;
    const SIDE_MULT: usize = PATCH_SIZE * POOLING_KERNEL_SIZE; // 48

    let (image_block, pixel_values) = if args.images.is_empty() {
        (String::new(), None)
    } else {
        if let Some(sz) = args.square_image_size {
            if sz % SIDE_MULT != 0 || sz == 0 {
                anyhow::bail!(
                    "--square-image-size ({sz}) must be a positive multiple of {SIDE_MULT}",
                );
            }
        }
        let mut tensors = Vec::with_capacity(args.images.len());
        let mut blocks = String::new();
        for path in &args.images {
            // Decode at full resolution to learn aspect ratio, then compute the
            // resize target that fits within the soft-token budget.
            let img = image::ImageReader::open(path)?
                .decode()
                .map_err(candle::Error::wrap)?;
            let (orig_h, orig_w) = (img.height() as usize, img.width() as usize);
            let (target_h, target_w) = match args.square_image_size {
                Some(sz) => (sz, sz),
                None => aspect_preserving_size(orig_h, orig_w, args.max_soft_tokens),
            };
            let patches_per_image = (target_h / PATCH_SIZE) * (target_w / PATCH_SIZE);
            let num_soft_tokens = patches_per_image / (POOLING_KERNEL_SIZE * POOLING_KERNEL_SIZE);

            blocks.push_str("<|image>");
            for _ in 0..num_soft_tokens {
                blocks.push_str("<|image|>");
            }
            blocks.push_str("<image|>");

            let resized = img.resize_exact(
                target_w as u32,
                target_h as u32,
                image::imageops::FilterType::Triangle,
            );
            let rgb = resized.to_rgb8();
            let data = rgb.into_raw();
            let chw = Tensor::from_vec(data, (target_h, target_w, 3), &Device::Cpu)?
                .permute((2, 0, 1))?
                .to_device(&device)?;
            // Rescale to [0, 1]; the gemma-4 image processor uses identity
            // mean/std so no further normalization is applied.
            let img = (chw.to_dtype(dtype)? / 255.0)?.unsqueeze(0)?;
            tensors.push(img);
        }
        (blocks, Some(tensors))
    };

    let prompt = if args.raw_prompt {
        args.prompt.clone()
    } else {
        format!(
            "<|turn>user\n{image_block}{}<turn|>\n<|turn>model\n",
            args.prompt
        )
    };
    pipeline.run(&prompt, args.sample_len, pixel_values)?;
    Ok(())
}
