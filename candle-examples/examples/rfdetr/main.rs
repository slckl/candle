mod config;
mod model;

use candle::DType;
use candle_nn::VarBuilder;
use clap::{Parser, ValueEnum};

use crate::config::RfDetrConfig;

/// RF-DETR model variants
#[derive(Clone, Copy, ValueEnum, Debug)]
pub enum Which {
    Nano,
    Small,
    Medium,
    Base,
    Large,
    LargeDeprecated,
    SegPreview,
    SegNano,
    SegSmall,
    SegMedium,
    SegLarge,
    SegXLarge,
    Seg2XLarge,
}

impl Which {
    pub fn config(&self) -> RfDetrConfig {
        match self {
            Which::Nano => RfDetrConfig::nano(),
            Which::Small => RfDetrConfig::small(),
            Which::Medium => RfDetrConfig::medium(),
            Which::Base => RfDetrConfig::base(),
            Which::Large => RfDetrConfig::large(),
            Which::LargeDeprecated => RfDetrConfig::large_deprecated(),
            Which::SegPreview => RfDetrConfig::seg_preview(),
            Which::SegNano => RfDetrConfig::seg_nano(),
            Which::SegSmall => RfDetrConfig::seg_small(),
            Which::SegMedium => RfDetrConfig::seg_medium(),
            Which::SegLarge => RfDetrConfig::seg_large(),
            Which::SegXLarge => RfDetrConfig::seg_xlarge(),
            Which::Seg2XLarge => RfDetrConfig::seg_2xlarge(),
        }
    }

    fn default_weights(&self) -> &'static str {
        match self {
            Which::Nano => "rfdetr-nano.safetensors",
            Which::Small => "rfdetr-small.safetensors",
            Which::Medium => "rfdetr-medium.safetensors",
            Which::Base => "rfdetr-base.safetensors",
            Which::Large => "rfdetr-large.safetensors",
            Which::LargeDeprecated => "rfdetr-large-deprecated.safetensors",
            Which::SegPreview => "rfdetr-seg-preview.safetensors",
            Which::SegNano => "rfdetr-seg-nano.safetensors",
            Which::SegSmall => "rfdetr-seg-small.safetensors",
            Which::SegMedium => "rfdetr-seg-medium.safetensors",
            Which::SegLarge => "rfdetr-seg-large.safetensors",
            Which::SegXLarge => "rfdetr-seg-xlarge.safetensors",
            Which::Seg2XLarge => "rfdetr-seg-2xlarge.safetensors",
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Model weights, in safetensors format.
    #[arg(long)]
    model: Option<String>,

    /// Which model variant to use.
    #[arg(long, value_enum, default_value_t = Which::Nano)]
    which: Which,

    images: Vec<String>,

    /// Threshold for the model confidence level.
    #[arg(long, default_value_t = 0.5)]
    confidence_threshold: f32,

    /// The size for the legend, 0 means no legend.
    #[arg(long, default_value_t = 20)]
    legend_size: u32,
}

impl Args {
    fn model(&self) -> anyhow::Result<std::path::PathBuf> {
        let path = match &self.model {
            Some(model) => std::path::PathBuf::from(model),
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("slckl/candle-rf-detr".to_string());
                let filename = self.which.default_weights();
                api.get(filename)?
            }
        };
        Ok(path)
    }
}

pub fn main() -> anyhow::Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    // Load model.
    let device = candle_examples::device(args.cpu)?;
    let model_path = args.model()?;
    let config = args.which.config();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };
    let model = RfDetr::load(vb, config)?;

    // TODO inference loop on images.

    Ok(())
}
