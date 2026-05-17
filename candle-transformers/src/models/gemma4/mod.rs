//! Gemma 4 multimodal model (text + vision + audio).
//!
//! See:
//! - [Google Blog](https://blog.google/technology/developers/gemma-4/)

pub mod audio;
pub mod config;
pub mod multimodal_embedding;
pub mod text;
pub mod vision;

use candle::{DType, Result, Tensor, D};

use config::Gemma4Config;
use multimodal_embedding::MultimodalEmbedder;
use text::TextModel;
use vision::VisionTower;

pub use audio::AudioModel;
pub use config::{Gemma4AudioConfig, Gemma4TextConfig, Gemma4VisionConfig};

/// Full Gemma4 multimodal model.
pub struct Model {
    pub language_model: TextModel,
    pub vision_tower: VisionTower,
    pub embed_vision: MultimodalEmbedder,
    pub audio_tower: Option<AudioModel>,
    pub embed_audio: Option<MultimodalEmbedder>,
    pub cfg: Gemma4Config,
}

impl Model {
    pub fn new(cfg: &Gemma4Config, vb: candle_nn::VarBuilder) -> Result<Self> {
        let vb = vb.pp("model");

        let vision_tower = VisionTower::new(&cfg.vision_config, vb.pp("vision_tower"))?;

        let vis_hidden = cfg.vision_config.hidden_size;
        let text_hidden = cfg.text_config.hidden_size;
        let embed_vision = MultimodalEmbedder::new(
            vis_hidden,
            text_hidden,
            cfg.vision_config.rms_norm_eps,
            vb.pp("embed_vision"),
        )?;

        let (audio_tower, embed_audio) = if let Some(ref audio_cfg) = cfg.audio_config {
            let tower = AudioModel::new(audio_cfg, vb.pp("audio_tower"))?;
            let audio_hidden = audio_cfg.output_proj_dims.unwrap_or(audio_cfg.hidden_size);
            let embed = MultimodalEmbedder::new(
                audio_hidden,
                text_hidden,
                audio_cfg.rms_norm_eps,
                vb.pp("embed_audio"),
            )?;
            (Some(tower), Some(embed))
        } else {
            (None, None)
        };

        let language_model = TextModel::new(&cfg.text_config, vb.pp("language_model"))?;

        Ok(Self {
            language_model,
            vision_tower,
            embed_vision,
            audio_tower,
            embed_audio,
            cfg: cfg.clone(),
        })
    }

    /// Text-only forward pass.
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.language_model.forward(input_ids, seqlen_offset)
    }

    /// Forward with multimodal inputs.
    ///
    /// `pixel_values`: optional batch of images, each `(1, C, H, W)` in [0, 1].
    /// `audio_mel`: optional `(batch, time, mel_bins)` mel spectrogram.
    /// `audio_mel_mask`: optional `(batch, time)` mask (1.0 = padding).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_multimodal(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&[Tensor]>,
        audio_mel: Option<&Tensor>,
        audio_mel_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mut input_embeds = self.language_model.embed_tokens(input_ids)?;

        // ── Vision embedding injection ──────────────────────────────────
        if let Some(pixel_values) = pixel_values {
            let vision_features = self.vision_tower.forward(pixel_values)?;
            let image_embeds = self
                .embed_vision
                .forward(&vision_features)?
                .to_dtype(input_embeds.dtype())?
                .squeeze(0)?;
            input_embeds =
                scatter_at_mask(&input_embeds, input_ids, self.cfg.image_token_id, &image_embeds)?;
        }

        // ── Audio embedding injection ───────────────────────────────────
        if let (
            Some(audio_mel),
            Some(audio_mel_mask),
            Some(ref audio_tower),
            Some(ref embed_audio),
        ) = (
            audio_mel,
            audio_mel_mask,
            &self.audio_tower,
            &self.embed_audio,
        ) {
            let (audio_features, enc_mask) = audio_tower.forward(audio_mel, audio_mel_mask)?;
            // Filter valid frames: where enc_mask == 0
            let valid = enc_mask.eq(0.0)?;
            let batch = audio_features.dim(0)?;
            let mut all_feats = Vec::new();
            for b in 0..batch {
                let valid_b = valid.get(b)?;
                let valid_sum = valid_b
                    .to_dtype(DType::F32)?
                    .sum_all()?
                    .to_scalar::<f32>()? as usize;
                if valid_sum > 0 {
                    all_feats.push(audio_features.get(b)?.narrow(0, 0, valid_sum)?);
                }
            }
            if !all_feats.is_empty() {
                let audio_feats = Tensor::cat(&all_feats, 0)?.unsqueeze(0)?;
                let audio_embeds = embed_audio
                    .forward(&audio_feats)?
                    .to_dtype(input_embeds.dtype())?
                    .squeeze(0)?;
                input_embeds = scatter_at_mask(
                    &input_embeds,
                    input_ids,
                    self.cfg.audio_token_id,
                    &audio_embeds,
                )?;
            }
        }

        self.language_model
            .forward_embeds(input_ids, &input_embeds, seqlen_offset)
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache()
    }
}

/// Replace `input_embeds` at every position where `input_ids == placeholder_id`
/// with the next row of `embeds`, in order. Returns the same shape as
/// `input_embeds` (`B, L, H`). Assumes batch size 1.
///
/// Implemented as a single masked `index_select` so it stays on-device: a
/// cumulative sum of the boolean mask gives each masked position its rank,
/// which is used as an `index_select` index into the modality embeddings.
fn scatter_at_mask(
    input_embeds: &Tensor,
    input_ids: &Tensor,
    placeholder_id: usize,
    embeds: &Tensor,
) -> Result<Tensor> {
    let (b_sz, seq_len) = input_ids.dims2()?;
    if b_sz != 1 {
        candle::bail!("gemma4 multimodal scatter only supports batch size 1");
    }
    let dtype = input_embeds.dtype();
    let device = input_embeds.device();

    // (1, L) boolean mask -> (L,) float.
    let mask = input_ids
        .to_dtype(DType::F32)?
        .eq(placeholder_id as f64)?
        .to_dtype(DType::F32)?
        .squeeze(0)?;
    // For position i with mask=1, the embedding row to pull is
    // `cumsum(mask)[i] - 1` (count of mask=1 strictly before i). For mask=0
    // positions we just clamp the index to a valid value; the result is
    // discarded by the mask multiplication below.
    let cum = mask.cumsum(0)?;
    let m = embeds.dim(0)? as f64;
    let idx = (cum - 1.0)?.clamp(0f64, m - 1.0)?.to_dtype(DType::U32)?;
    let gathered = embeds.index_select(&idx, 0)?.unsqueeze(0)?;

    let mask_3d = mask.unsqueeze(0)?.unsqueeze(D::Minus1)?.to_dtype(dtype)?;
    let one = Tensor::ones((1, seq_len, 1), dtype, device)?;
    let inv_mask = (&one - &mask_3d)?;
    mask_3d.broadcast_mul(&gathered)? + inv_mask.broadcast_mul(input_embeds)?
}
