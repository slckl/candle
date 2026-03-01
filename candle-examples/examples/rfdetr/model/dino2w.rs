//! DINOv2 with Windowed Attention Backbone for RF-DETR.
//! Supports 2 variants: DINOv2-Small (384 hidden dim) and DINOv2-Base (768 hidden dim).

use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder};

/// Configuration for the DINOv2 backbone
#[derive(Debug, Clone)]
pub struct Dinov2Config {
    /// Hidden dimension size (384 for small, 768 for base)
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// MLP intermediate size ratio (typically 4)
    pub mlp_ratio: usize,
    /// Layer normalization epsilon
    pub layer_norm_eps: f64,
    /// Input image size (used for position embedding interpolation)
    pub image_size: usize,
    /// Patch size for the patch embedding convolution
    pub patch_size: usize,
    /// Number of input channels (typically 3 for RGB)
    pub num_channels: usize,
    /// Number of register tokens (0 for RF-DETR small)
    pub num_register_tokens: usize,
    /// Number of windows for windowed attention
    pub num_windows: usize,
    /// Layer indices that use windowed attention (others use full attention)
    pub window_block_indexes: Vec<usize>,
    /// Output feature stage names (e.g., ["stage3", "stage6", "stage9", "stage12"])
    pub out_features: Vec<String>,
    /// Whether to reshape hidden states to 4D (B, C, H, W)
    pub reshape_hidden_states: bool,
}

impl Dinov2Config {
    /// Create configuration for DINOv2-Small with windowed attention
    /// As used in RF-DETR small/medium/nano models
    pub fn small_windowed(
        image_size: usize,
        patch_size: usize,
        num_windows: usize,
        out_feature_indexes: &[usize],
    ) -> Self {
        // Window block indexes calculation matches Python:
        // window_block_indexes = set(range(out_feature_indexes[-1] + 1))
        // window_block_indexes.difference_update(out_feature_indexes)
        // This uses the 1-indexed stage numbers directly
        let max_stage = *out_feature_indexes.iter().max().unwrap_or(&12);
        let all_stages: std::collections::HashSet<usize> = (0..=max_stage).collect();
        let out_stages: std::collections::HashSet<usize> =
            out_feature_indexes.iter().copied().collect();
        let window_block_indexes: Vec<usize> =
            all_stages.difference(&out_stages).copied().collect();

        let out_features = out_feature_indexes
            .iter()
            .map(|&i| format!("stage{}", i))
            .collect();

        Self {
            hidden_size: 384,
            num_hidden_layers: 12,
            num_attention_heads: 6,
            mlp_ratio: 4,
            layer_norm_eps: 1e-6,
            image_size,
            patch_size,
            num_channels: 3,
            num_register_tokens: 0,
            num_windows,
            window_block_indexes,
            out_features,
            reshape_hidden_states: true,
        }
    }

    /// Create configuration for DINOv2-Base with windowed attention
    /// As used in RF-DETR base/large models
    pub fn base_windowed(
        image_size: usize,
        patch_size: usize,
        num_windows: usize,
        out_feature_indexes: &[usize],
    ) -> Self {
        // Window block indexes calculation matches Python
        let max_stage = *out_feature_indexes.iter().max().unwrap_or(&12);
        let all_stages: std::collections::HashSet<usize> = (0..=max_stage).collect();
        let out_stages: std::collections::HashSet<usize> =
            out_feature_indexes.iter().copied().collect();
        let window_block_indexes: Vec<usize> =
            all_stages.difference(&out_stages).copied().collect();

        let out_features = out_feature_indexes
            .iter()
            .map(|&i| format!("stage{}", i))
            .collect();

        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            mlp_ratio: 4,
            layer_norm_eps: 1e-6,
            image_size,
            patch_size,
            num_channels: 3,
            num_register_tokens: 0, // RF-DETR doesn't use register tokens
            num_windows,
            window_block_indexes,
            out_features,
            reshape_hidden_states: true,
        }
    }

    /// Check if a layer index uses windowed attention
    pub fn is_windowed_layer(&self, layer_idx: usize) -> bool {
        self.window_block_indexes.contains(&layer_idx)
    }
}

/// Patch embeddings using a Conv2d projection
pub struct PatchEmbeddings {
    projection: Conv2d,
}

impl PatchEmbeddings {
    pub fn load(vb: VarBuilder, config: &Dinov2Config) -> Result<Self> {
        let conv_config = Conv2dConfig {
            stride: config.patch_size,
            ..Default::default()
        };
        let projection = candle_nn::conv2d(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            conv_config,
            vb.pp("projection"),
        )?;
        Ok(Self { projection })
    }
}

impl Module for PatchEmbeddings {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // pixel_values: [batch_size, num_channels, height, width]
        // output: [batch_size, num_patches, hidden_size]
        let embeddings = self.projection.forward(pixel_values)?;
        // [batch_size, hidden_size, h_patches, w_patches] -> [batch_size, hidden_size, num_patches]
        let embeddings = embeddings.flatten_from(2)?;
        // [batch_size, hidden_size, num_patches] -> [batch_size, num_patches, hidden_size]
        embeddings.transpose(1, 2)
    }
}

/// Embeddings module combining patch embeddings, position embeddings, and special tokens
pub struct Embeddings {
    cls_token: Tensor,
    position_embeddings: Tensor,
    patch_embeddings: PatchEmbeddings,
    register_tokens: Option<Tensor>,
    config: Dinov2Config,
}

impl Embeddings {
    pub fn load(vb: VarBuilder, config: &Dinov2Config) -> Result<Self> {
        let cls_token = vb.get((1, 1, config.hidden_size), "cls_token")?;

        // Position embeddings: [1, num_patches + 1, hidden_size]
        // For the stored weights, num_patches is based on the training image size
        let position_embeddings = vb.get_with_hints_dtype(
            (
                1,
                (config.image_size / config.patch_size).pow(2) + 1,
                config.hidden_size,
            ),
            "position_embeddings",
            Default::default(),
            DType::F32,
        )?;

        let patch_embeddings = PatchEmbeddings::load(vb.pp("patch_embeddings"), config)?;

        let register_tokens = if config.num_register_tokens > 0 {
            Some(vb.get(
                (1, config.num_register_tokens, config.hidden_size),
                "register_tokens",
            )?)
        } else {
            None
        };

        Ok(Self {
            cls_token,
            position_embeddings,
            patch_embeddings,
            register_tokens,
            config: config.clone(),
        })
    }

    /// Interpolate position encodings for different input sizes
    fn interpolate_pos_encoding(
        &self,
        embeddings: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let num_patches = embeddings.dim(1)? - 1;
        let num_positions = self.position_embeddings.dim(1)? - 1;

        let target_h = height / self.config.patch_size;
        let target_w = width / self.config.patch_size;

        // If dimensions match, return as is
        if num_patches == num_positions && target_h == target_w {
            return Ok(self.position_embeddings.clone());
        }

        // Separate class token and patch embeddings
        let class_pos_embed = self.position_embeddings.i((.., ..1, ..))?;
        let patch_pos_embed = self.position_embeddings.i((.., 1.., ..))?;

        let dim = embeddings.dim(D::Minus1)?;
        let sqrt_num_positions = (num_positions as f64).sqrt() as usize;

        // Reshape for interpolation: [1, sqrt_n, sqrt_n, dim] -> [1, dim, sqrt_n, sqrt_n]
        let patch_pos_embed =
            patch_pos_embed.reshape((1, sqrt_num_positions, sqrt_num_positions, dim))?;
        let patch_pos_embed = patch_pos_embed.permute((0, 3, 1, 2))?;

        // Interpolate using bilinear (bicubic not available in candle, bilinear is close enough)
        let patch_pos_embed = patch_pos_embed.upsample_nearest2d(target_h, target_w)?;

        // Reshape back: [1, dim, h, w] -> [1, h*w, dim]
        let patch_pos_embed = patch_pos_embed.permute((0, 2, 3, 1))?;
        let patch_pos_embed = patch_pos_embed.reshape((1, target_h * target_w, dim))?;

        // Concatenate class token and interpolated patch embeddings
        Tensor::cat(&[&class_pos_embed, &patch_pos_embed], 1)
    }

    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let (batch_size, _, height, width) = pixel_values.dims4()?;

        // Get patch embeddings
        let embeddings = self.patch_embeddings.forward(pixel_values)?;

        // Expand cls token for batch
        let cls_tokens = self
            .cls_token
            .broadcast_as((batch_size, 1, self.config.hidden_size))?;

        // Concatenate cls token with patch embeddings
        let embeddings = Tensor::cat(&[&cls_tokens, &embeddings], 1)?;

        // Add position embeddings
        let pos_embed = self.interpolate_pos_encoding(&embeddings, height, width)?;
        let mut embeddings = embeddings.broadcast_add(&pos_embed)?;

        // Apply windowing if num_windows > 1
        if self.config.num_windows > 1 {
            let num_h_patches = height / self.config.patch_size;
            let num_w_patches = width / self.config.patch_size;

            // Separate cls token and patch tokens
            let cls_token_with_pos = embeddings.i((.., ..1, ..))?;
            let patch_tokens = embeddings.i((.., 1.., ..))?;

            // Reshape patch tokens for windowing
            // [B, H*W, C] -> [B, H, W, C]
            let patch_tokens =
                patch_tokens.reshape((batch_size, num_h_patches, num_w_patches, ()))?;

            let num_windows = self.config.num_windows;
            let num_h_per_window = num_h_patches / num_windows;
            let num_w_per_window = num_w_patches / num_windows;

            // Following Python implementation exactly:
            // windowed_pixel_tokens = pixel_tokens_with_pos_embed.reshape(
            //     batch_size * num_windows, num_h_patches_per_window, num_windows, num_h_patches_per_window, -1)
            // Note: Python uses num_h_patches_per_window for both h and w dimensions in reshape
            // This is actually: [B*num_win, h_per_win, num_win, w_per_win, C]
            let patch_tokens = patch_tokens.reshape((
                batch_size * num_windows,
                num_h_per_window,
                num_windows,
                num_w_per_window,
                self.config.hidden_size,
            ))?;

            // windowed_pixel_tokens = windowed_pixel_tokens.permute(0, 2, 1, 3, 4)
            // Result: [B*num_win, num_win, h_per_win, w_per_win, C]
            let patch_tokens = patch_tokens.permute((0, 2, 1, 3, 4))?;

            // windowed_pixel_tokens = windowed_pixel_tokens.reshape(
            //     batch_size * num_windows ** 2, num_h_patches_per_window * num_w_patches_per_window, -1)
            let num_windows_sq = num_windows * num_windows;
            let tokens_per_window = num_h_per_window * num_w_per_window;
            let patch_tokens = patch_tokens.reshape((
                batch_size * num_windows_sq,
                tokens_per_window,
                self.config.hidden_size,
            ))?;

            // windowed_cls_token_with_pos_embed = cls_token_with_pos_embed.repeat(num_windows ** 2, 1, 1)
            let cls_tokens = cls_token_with_pos.repeat((num_windows_sq, 1, 1))?;

            // embeddings = torch.cat((windowed_cls_token_with_pos_embed, windowed_pixel_tokens), dim=1)
            embeddings = Tensor::cat(&[&cls_tokens, &patch_tokens], 1)?;
        }

        // Add register tokens if present
        if let Some(ref register_tokens) = self.register_tokens {
            let current_batch = embeddings.dim(0)?;
            let reg_tokens = register_tokens.broadcast_as((
                current_batch,
                self.config.num_register_tokens,
                self.config.hidden_size,
            ))?;
            // Insert register tokens after cls token
            let cls_part = embeddings.i((.., ..1, ..))?;
            let rest = embeddings.i((.., 1.., ..))?;
            embeddings = Tensor::cat(&[&cls_part, &reg_tokens, &rest], 1)?;
        }

        Ok(embeddings)
    }
}

#[derive(Debug)]
pub struct Attention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    scale: f64,
}

impl Attention {
    pub fn load(vb: VarBuilder, config: &Dinov2Config) -> Result<Self> {
        let dim = config.hidden_size;
        let num_heads = config.num_attention_heads;

        // Load separate q/k/v weights and fuse them into a single qkv to match dino2 reference candle impl.
        let attn_vb = vb.pp("attention");
        let q_weight = attn_vb.pp("query").get((dim, dim), "weight")?;
        let k_weight = attn_vb.pp("key").get((dim, dim), "weight")?;
        let v_weight = attn_vb.pp("value").get((dim, dim), "weight")?;
        let qkv_weight = Tensor::cat(&[&q_weight, &k_weight, &v_weight], 0)?;

        let q_bias = attn_vb.pp("query").get(dim, "bias")?;
        let k_bias = attn_vb.pp("key").get(dim, "bias")?;
        let v_bias = attn_vb.pp("value").get(dim, "bias")?;
        let qkv_bias = Tensor::cat(&[&q_bias, &k_bias, &v_bias], 0)?;

        let qkv = Linear::new(qkv_weight, Some(qkv_bias));

        let proj = candle_nn::linear(dim, dim, vb.pp("output.dense"))?;
        let scale = 1. / ((dim / num_heads) as f64).sqrt();

        Ok(Self {
            qkv,
            proj,
            num_heads,
            scale,
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, n, 3, self.num_heads, c / self.num_heads))?
            .transpose(1, 2)? // b,3,n,h,d
            .transpose(0, 1)? // 3,b,n,h,d
            .transpose(2, 3)?; // 3,b,h,n,d
        let q = (qkv.i(0)? * self.scale)?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;
        let attn = candle_nn::ops::softmax(&q.matmul(&k.t()?)?, D::Minus1)?;
        let attn = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, n, c))?;
        self.proj.forward(&attn)
    }
}

/// Layer scale: learnable per-channel scaling
#[derive(Debug)]
pub struct LayerScale {
    lambda: Tensor,
}

impl LayerScale {
    pub fn load(vb: VarBuilder, dim: usize) -> Result<Self> {
        let lambda = vb.get(dim, "lambda1")?;
        Ok(Self { lambda })
    }
}

impl Module for LayerScale {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&self.lambda)
    }
}

#[derive(Debug)]
pub struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    pub fn load(vb: VarBuilder, config: &Dinov2Config) -> Result<Self> {
        let hidden_features = config.hidden_size * config.mlp_ratio;
        let fc1 = candle_nn::linear(config.hidden_size, hidden_features, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(hidden_features, config.hidden_size, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        // TODO do we need gelu_erf(), won't just gelu() be enough?
        let xs = xs.gelu_erf()?;
        self.fc2.forward(&xs)
    }
}

/// Single transformer layer/block
#[derive(Debug)]
pub struct Layer {
    norm1: LayerNorm,
    attn: Attention,
    ls1: LayerScale,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
    num_windows: usize,
}

impl Layer {
    pub fn load(vb: VarBuilder, config: &Dinov2Config) -> Result<Self> {
        let norm1 = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("norm1"))?;
        let attention = Attention::load(vb.pp("attention"), config)?;
        let layer_scale1 = LayerScale::load(vb.pp("layer_scale1"), config.hidden_size)?;
        let norm2 = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("norm2"))?;
        let mlp = Mlp::load(vb.pp("mlp"), config)?;
        let layer_scale2 = LayerScale::load(vb.pp("layer_scale2"), config.hidden_size)?;

        Ok(Self {
            norm1,
            attn: attention,
            ls1: layer_scale1,
            norm2,
            mlp,
            ls2: layer_scale2,
            num_windows: config.num_windows,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, run_full_attention: bool) -> Result<Tensor> {
        let shortcut = hidden_states.clone();

        // For full attention layers, merge windows before attention
        // Save the windowed dimensions for reshaping back later
        let (windowed_b, windowed_hw, windowed_c) = hidden_states.dims3()?;
        let hidden_states = if run_full_attention && self.num_windows > 1 {
            let num_windows_sq = self.num_windows * self.num_windows;
            hidden_states.reshape((
                windowed_b / num_windows_sq,
                num_windows_sq * windowed_hw,
                windowed_c,
            ))?
        } else {
            hidden_states.clone()
        };

        // Self-attention with pre-norm
        let normed = self.norm1.forward(&hidden_states)?;
        let mut attention_output = self.attn.forward(&normed)?;

        // For full attention layers, split back to windows after attention
        // Use the merged dimensions (from hidden_states after merge) for the split
        if run_full_attention && self.num_windows > 1 {
            let (merged_b, merged_hw, c) = hidden_states.dims3()?;
            let num_windows_sq = self.num_windows * self.num_windows;
            // Split back: [B, num_win^2 * hw_per_win, C] -> [B * num_win^2, hw_per_win, C]
            attention_output = attention_output.reshape((
                merged_b * num_windows_sq,
                merged_hw / num_windows_sq,
                c,
            ))?;
        }

        // Layer scale and residual
        let attention_output = self.ls1.forward(&attention_output)?;
        let hidden_states = (shortcut + attention_output)?;

        // MLP with pre-norm
        let shortcut = hidden_states.clone();
        let normed = self.norm2.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        let mlp_output = self.ls2.forward(&mlp_output)?;

        shortcut + mlp_output
    }
}

/// Transformer encoder stack
pub struct Encoder {
    layers: Vec<Layer>,
    config: Dinov2Config,
}

impl Encoder {
    pub fn load(vb: VarBuilder, config: &Dinov2Config) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = Layer::load(vb.pp(format!("layer.{}", i)), config)?;
            layers.push(layer);
        }
        Ok(Self {
            layers,
            config: config.clone(),
        })
    }

    /// Forward pass returning hidden states at all layers
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Vec<Tensor>> {
        let mut all_hidden_states = Vec::with_capacity(self.config.num_hidden_layers + 1);
        all_hidden_states.push(hidden_states.clone());

        let mut hidden_states = hidden_states.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let run_full_attention = !self.config.is_windowed_layer(i);
            hidden_states = layer.forward(&hidden_states, run_full_attention)?;
            all_hidden_states.push(hidden_states.clone());
        }

        Ok(all_hidden_states)
    }
}

/// Complete DINOv2 backbone with windowed attention
pub struct Dinov2Backbone {
    embeddings: Embeddings,
    encoder: Encoder,
    layernorm: LayerNorm,
    config: Dinov2Config,
}

impl Dinov2Backbone {
    pub fn load(vb: VarBuilder, config: &Dinov2Config) -> Result<Self> {
        let embeddings = Embeddings::load(vb.pp("embeddings"), config)?;
        let encoder = Encoder::load(vb.pp("encoder"), config)?;
        let layernorm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("layernorm"),
        )?;

        Ok(Self {
            embeddings,
            encoder,
            layernorm,
            config: config.clone(),
        })
    }

    /// Forward pass returning feature maps at specified output stages
    ///
    /// Returns a Vec of tensors with shape [B, C, H, W] for each output stage
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Vec<Tensor>> {
        let (batch_size, _, height, width) = pixel_values.dims4()?;

        // Get embeddings (potentially windowed)
        let embedding_output = self.embeddings.forward(pixel_values)?;

        // Run through encoder, collecting all hidden states
        let all_hidden_states = self.encoder.forward(&embedding_output)?;

        // Extract feature maps at output stages
        let mut feature_maps = Vec::new();
        let stage_names: Vec<String> = std::iter::once("stem".to_string())
            .chain((1..=self.config.num_hidden_layers).map(|i| format!("stage{}", i)))
            .collect();

        for (stage_name, hidden_state) in stage_names.iter().zip(all_hidden_states.iter()) {
            if !self.config.out_features.contains(stage_name) {
                continue;
            }

            let mut hidden_state = hidden_state.clone();
            hidden_state = self.layernorm.forward(&hidden_state)?;

            if self.config.reshape_hidden_states {
                // Remove cls token (and register tokens if present)
                let skip_tokens = 1 + self.config.num_register_tokens;
                hidden_state = hidden_state.i((.., skip_tokens.., ..))?;

                let num_h_patches = height / self.config.patch_size;
                let num_w_patches = width / self.config.patch_size;

                // Undo windowing if needed
                if self.config.num_windows > 1 {
                    let num_windows = self.config.num_windows;
                    let num_windows_sq = num_windows * num_windows;
                    let (b, hw, c) = hidden_state.dims3()?;

                    let num_h_per_window = num_h_patches / num_windows;
                    let num_w_per_window = num_w_patches / num_windows;

                    // Merge windows back: [B*num_win^2, h_per_win*w_per_win, C]
                    // -> [B, num_win^2*h_per_win*w_per_win, C]
                    hidden_state =
                        hidden_state.reshape((b / num_windows_sq, num_windows_sq * hw, c))?;

                    // Reshape to spatial: [B, num_win, num_win, h_per_win, w_per_win, C]
                    hidden_state = hidden_state.reshape((
                        batch_size,
                        num_windows,
                        num_windows,
                        num_h_per_window,
                        num_w_per_window,
                        c,
                    ))?;

                    // Permute to [B, num_win, h_per_win, num_win, w_per_win, C]
                    hidden_state = hidden_state.permute((0, 1, 3, 2, 4, 5))?;

                    // Reshape to [B, H, W, C]
                    hidden_state =
                        hidden_state.reshape((batch_size, num_h_patches, num_w_patches, c))?;
                } else {
                    // Simple reshape without windowing
                    let c = hidden_state.dim(D::Minus1)?;
                    hidden_state =
                        hidden_state.reshape((batch_size, num_h_patches, num_w_patches, c))?;
                }

                // [B, H, W, C] -> [B, C, H, W]
                hidden_state = hidden_state.permute((0, 3, 1, 2))?.contiguous()?;
            }

            feature_maps.push(hidden_state);
        }

        Ok(feature_maps)
    }
}
