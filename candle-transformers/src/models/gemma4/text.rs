//! Gemma 4 text decoder.
//!
//! Implements the language-model portion of Google's Gemma 4 multimodal
//! checkpoints (e.g. `google/gemma-4-E2B-it`, `gemma-4-E4B-it`). The key
//! differences from Gemma 1/2/3 are:
//!
//! - `RmsNorm` scales by `weight` directly (no `+1` offset).
//! - Attention uses `scaling = 1.0` (Q/K norms produce the effective scale).
//! - Per-Layer Embeddings (PLE): an auxiliary embedding/projection stream
//!   feeds a small residual into each decoder layer.
//! - A learned per-layer `layer_scalar` multiplies the layer output.
//! - Layers in the KV-sharing range use a double-wide MLP when
//!   `use_double_wide_mlp` is set.

use std::sync::Arc;

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_b as linear_bias, Activation, Linear, VarBuilder};

use super::config::Gemma4TextConfig;

// ── RmsNorm (Gemma4: scales by `weight`, no +1 offset) ──────────────────────

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Option<Tensor>,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self {
            weight: Some(weight),
            eps,
        })
    }

    /// RMS normalization without a learned scale (used for the V projection).
    fn new_no_scale(eps: f64) -> Self {
        Self { weight: None, eps }
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let x_f = x.to_dtype(internal_dtype)?;
        let mean_sq = x_f.sqr()?.mean_keepdim(D::Minus1)?;
        let rms = (mean_sq + self.eps)?.sqrt()?;
        let x_normed = x_f.broadcast_div(&rms)?;
        let out = match &self.weight {
            Some(w) => x_normed.broadcast_mul(&w.to_dtype(internal_dtype)?)?,
            None => x_normed,
        };
        out.to_dtype(x_dtype)
    }
}

// ── RotaryEmbedding (standard, used for sliding/local layers) ───────────────

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── ProportionalRotaryEmbedding (for global / full-attention layers) ────────
//
// Rotates only the first `partial_rotary_factor * head_dim` channels and
// leaves the rest as identity (cos=1, sin=0).

#[derive(Debug, Clone)]
struct ProportionalRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl ProportionalRotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        partial_rotary_factor: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let rope_angles = (partial_rotary_factor * head_dim as f64 / 2.0) as usize;
        let half_dim = head_dim / 2;

        let mut inv_freq_vec = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq_vec.push(1f32 / (rope_theta as f32).powf((2 * i) as f32 / head_dim as f32));
        }
        inv_freq_vec.extend(std::iter::repeat_n(0f32, half_dim - rope_angles));

        let inv_freq = Tensor::from_vec(inv_freq_vec, (1, half_dim), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── MLP ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        act: Activation,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate_proj = linear_bias(hidden_size, intermediate_size, bias, vb.pp("gate_proj"))?;
        let up_proj = linear_bias(hidden_size, intermediate_size, bias, vb.pp("up_proj"))?;
        let down_proj = linear_bias(intermediate_size, hidden_size, bias, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// ── Flash attention ─────────────────────────────────────────────────────────

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

// ── KvCache ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum KvCache {
    Normal(candle_nn::kv_cache::KvCache),
    Rotating(candle_nn::kv_cache::RotatingKvCache),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LayerType {
    Sliding,
    Full,
}

/// Cached K/V (already RoPE'd, at `num_kv_heads`) from the most recent
/// non-shared layer of each `LayerType`. Layers with `is_kv_shared = true`
/// borrow from here instead of computing fresh K/V projections.
#[derive(Debug, Clone, Default)]
pub struct SharedKvStates {
    sliding: Option<(Tensor, Tensor)>,
    full: Option<(Tensor, Tensor)>,
}

impl SharedKvStates {
    fn get(&self, layer_type: LayerType) -> Option<&(Tensor, Tensor)> {
        match layer_type {
            LayerType::Sliding => self.sliding.as_ref(),
            LayerType::Full => self.full.as_ref(),
        }
    }
    fn set(&mut self, layer_type: LayerType, k: Tensor, v: Tensor) {
        match layer_type {
            LayerType::Sliding => self.sliding = Some((k, v)),
            LayerType::Full => self.full = Some((k, v)),
        }
    }
}

// ── Attention ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Option<Linear>,
    v_proj: Option<Linear>,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: Option<RmsNorm>,
    v_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    is_sliding: bool,
    layer_type: LayerType,
    is_kv_shared_layer: bool,
    store_full_length_kv: bool,
    rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
    kv_cache: Option<KvCache>,
    use_flash_attn: bool,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let bias = cfg.attention_bias;
        let is_sliding = cfg.is_sliding(layer_idx);
        let layer_type = if is_sliding {
            LayerType::Sliding
        } else {
            LayerType::Full
        };

        let (head_dim, num_kv_heads) = if is_sliding {
            (cfg.head_dim, cfg.num_key_value_heads)
        } else {
            let global_kv = cfg
                .num_global_key_value_heads
                .unwrap_or(cfg.num_key_value_heads);
            (cfg.global_head_dim, global_kv)
        };

        let is_kv_shared_layer = cfg.is_kv_shared_layer(layer_idx);
        // The donor layer for `shared_kv_states[layer_type]` is the most
        // recent non-shared layer of the same type. Only that layer needs to
        // export its K/V; we precompute that boolean here.
        let store_full_length_kv = !is_kv_shared_layer && {
            let last_shared_donor = (0..cfg.first_kv_shared_layer_idx())
                .rev()
                .find(|&i| cfg.is_sliding(i) == is_sliding);
            last_shared_donor == Some(layer_idx)
        };

        let num_kv_groups = num_heads / num_kv_heads;
        let q_proj = linear_bias(hidden_sz, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let o_proj = linear_bias(num_heads * head_dim, hidden_sz, bias, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let v_norm = RmsNorm::new_no_scale(cfg.rms_norm_eps);

        // K/V projections and K-norm only exist (and are only useful) on
        // non-shared layers. Skip loading them for shared layers so the model
        // still works if a checkpoint truncates them.
        let (k_proj, v_proj, k_norm, kv_cache) = if is_kv_shared_layer {
            (None, None, None, None)
        } else {
            let k_proj = linear_bias(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?;
            let v_proj = linear_bias(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?;
            let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
            let cache = if is_sliding {
                KvCache::Rotating(candle_nn::kv_cache::RotatingKvCache::new(
                    2,
                    cfg.effective_sliding_window(),
                ))
            } else {
                KvCache::Normal(candle_nn::kv_cache::KvCache::new(
                    2,
                    cfg.max_position_embeddings,
                ))
            };
            (Some(k_proj), Some(v_proj), Some(k_norm), Some(cache))
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            is_sliding,
            layer_type,
            is_kv_shared_layer,
            store_full_length_kv,
            rotary_emb_global,
            rotary_emb_local,
            kv_cache,
            use_flash_attn: cfg.use_flash_attn,
        })
    }

    fn apply_rope(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        if self.is_sliding {
            self.rotary_emb_local.apply(q, k, seqlen_offset)
        } else {
            self.rotary_emb_global.apply(q, k, seqlen_offset)
        }
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        shared_kv: &mut SharedKvStates,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let mut q = self.q_proj.forward(xs)?;
        q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        q = self.q_norm.forward(&q)?;

        let (q, k, v) = if self.is_kv_shared_layer {
            // Shared layer: only compute Q. Borrow the donor layer's already
            // RoPE'd K/V cache. Apply RoPE to Q at the current position.
            let (k, v) = shared_kv
                .get(self.layer_type)
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "gemma4 shared kv layer {:?} requested but donor cache is empty",
                        self.layer_type
                    ))
                })?
                .clone();
            // Build a zero K of matching shape just to satisfy the rope helper,
            // then drop it — we only need the rotated Q.
            let zero_k = Tensor::zeros(
                (b_sz, self.num_kv_heads, q_len, self.head_dim),
                q.dtype(),
                q.device(),
            )?;
            let (q_rot, _) = self.apply_rope(&q, &zero_k, seqlen_offset)?;
            (q_rot, k, v)
        } else {
            let k_proj = self.k_proj.as_ref().expect("non-shared layer has k_proj");
            let v_proj = self.v_proj.as_ref().expect("non-shared layer has v_proj");
            let k_norm = self.k_norm.as_ref().expect("non-shared layer has k_norm");
            let mut k = k_proj.forward(xs)?;
            let v = v_proj.forward(xs)?;
            k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            k = k_norm.forward(&k)?;
            let v = self.v_norm.forward(&v)?;
            let (q, k) = self.apply_rope(&q, &k, seqlen_offset)?;
            let (k, v) = match self.kv_cache.as_mut().expect("non-shared has cache") {
                KvCache::Normal(cache) => cache.append(&k, &v)?,
                KvCache::Rotating(cache) => cache.append(&k, &v)?,
            };
            if self.store_full_length_kv {
                shared_kv.set(self.layer_type, k.clone(), v.clone());
            }
            (q, k, v)
        };

        let k = crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let mask = if self.is_sliding {
            sliding_attention_mask
        } else {
            attention_mask
        };

        // Gemma4 uses scaling = 1.0 (Q/K norms set the effective scale).
        let attn_output = if self.use_flash_attn {
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            flash_attn(&q, &k, &v, 1.0, mask.is_some())?.transpose(1, 2)?
        } else {
            let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
            let attn_weights = match mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v)?
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        if let Some(cache) = self.kv_cache.as_mut() {
            match cache {
                KvCache::Normal(c) => c.reset(),
                KvCache::Rotating(c) => c.reset(),
            }
        }
    }
}

// ── DecoderLayer ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    // Per-Layer Embedding (PLE) block.
    per_layer_input_gate: Linear,
    per_layer_projection: Linear,
    post_per_layer_input_norm: RmsNorm,
    act_fn: Activation,
    // Learned scalar that scales the layer output.
    layer_scalar: Tensor,
}

impl DecoderLayer {
    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb_global,
            rotary_emb_local,
            cfg,
            layer_idx,
            vb.pp("self_attn"),
        )?;
        let mlp = MLP::new(
            cfg.hidden_size,
            cfg.layer_intermediate_size(layer_idx),
            cfg.hidden_activation,
            false,
            vb.pp("mlp"),
        )?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        let per_layer_input_gate = linear_bias(
            cfg.hidden_size,
            cfg.hidden_size_per_layer_input,
            false,
            vb.pp("per_layer_input_gate"),
        )?;
        let per_layer_projection = linear_bias(
            cfg.hidden_size_per_layer_input,
            cfg.hidden_size,
            false,
            vb.pp("per_layer_projection"),
        )?;
        let post_per_layer_input_norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_per_layer_input_norm"),
        )?;
        let layer_scalar = vb.get(1, "layer_scalar")?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            act_fn: cfg.hidden_activation,
            layer_scalar,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        per_layer_input: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        shared_kv: &mut SharedKvStates,
    ) -> Result<Tensor> {
        // Standard pre-LN attention + MLP block.
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            sliding_attention_mask,
            seqlen_offset,
            shared_kv,
        )?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let mlp_out = xs.apply(&self.pre_feedforward_layernorm)?;
        let mlp_out = mlp_out.apply(&self.mlp)?;
        let mlp_out = mlp_out.apply(&self.post_feedforward_layernorm)?;
        let xs = (residual + mlp_out)?;

        // PLE residual: gate → activation → element-wise multiply by per-layer
        // input → projection back to hidden_size → norm → residual add.
        let residual = &xs;
        let ple = self.per_layer_input_gate.forward(&xs)?;
        let ple = ple.apply(&self.act_fn)?;
        let ple = ple.broadcast_mul(per_layer_input)?;
        let ple = self.per_layer_projection.forward(&ple)?;
        let ple = self.post_per_layer_input_norm.forward(&ple)?;
        let xs = (residual + ple)?;

        // Final learned scalar gain on the layer output.
        xs.broadcast_mul(&self.layer_scalar)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

// ── Causal mask ─────────────────────────────────────────────────────────────

fn prepare_decoder_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = if let Some(sliding_window) = sliding_window {
        (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect()
    } else {
        (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect()
    };
    let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)
}

// ── TextModel ───────────────────────────────────────────────────────────────
//
// `vb` should already point at the prefix where the language-model weights
// live. For the released multimodal checkpoints that is
// `model.language_model.*`; pass `root.pp("model").pp("language_model")` from
// `main.rs` (text-only path) or `vb.pp("language_model")` from inside the
// multimodal `Model::new` (which has already pp'd to `model`).
//
// `lm_head` is constructed from the tied `embed_tokens` weight matrix when
// `tie_word_embeddings` is true. Loading a separate `lm_head` for untied
// checkpoints is not supported here.

#[derive(Debug, Clone)]
pub struct TextModel {
    embed_tokens: candle_nn::Embedding,
    embed_tokens_per_layer: candle_nn::Embedding,
    per_layer_model_projection: Linear,
    per_layer_projection_norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    hidden_size_per_layer_input: usize,
    num_hidden_layers: usize,
    sliding_window: usize,
}

impl TextModel {
    pub fn new(cfg: &Gemma4TextConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;

        let embed_tokens_per_layer = candle_nn::embedding(
            cfg.vocab_size_per_layer_input,
            cfg.num_hidden_layers * cfg.hidden_size_per_layer_input,
            vb.pp("embed_tokens_per_layer"),
        )?;
        let per_layer_model_projection = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_hidden_layers * cfg.hidden_size_per_layer_input,
            vb.pp("per_layer_model_projection"),
        )?;
        let per_layer_projection_norm = RmsNorm::new(
            cfg.hidden_size_per_layer_input,
            cfg.rms_norm_eps,
            vb.pp("per_layer_projection_norm"),
        )?;

        let rotary_emb_global = Arc::new(ProportionalRotaryEmbedding::new(
            vb.dtype(),
            cfg.global_head_dim,
            cfg.rope_theta,
            cfg.partial_rotary_factor(),
            cfg.max_position_embeddings,
            vb.device(),
        )?);
        let rotary_emb_local = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            cfg.head_dim,
            cfg.rope_local_base_freq(),
            cfg.max_position_embeddings,
            vb.device(),
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                rotary_emb_global.clone(),
                rotary_emb_local.clone(),
                cfg,
                layer_idx,
                vb_l.pp(layer_idx),
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self {
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            hidden_size_per_layer_input: cfg.hidden_size_per_layer_input,
            num_hidden_layers: cfg.num_hidden_layers,
            sliding_window: cfg.sliding_window,
        })
    }

    fn create_attention_masks(
        &self,
        batch_size: usize,
        seq_len: usize,
        seqlen_offset: usize,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if seq_len <= 1 {
            return Ok((None, None));
        }
        let mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            None,
            self.dtype,
            &self.device,
        )?;
        let sliding_mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            Some(self.sliding_window),
            self.dtype,
            &self.device,
        )?;
        Ok((Some(mask), Some(sliding_mask)))
    }

    /// Per-Layer Embedding (PLE) inputs for the current batch.
    ///
    /// Returns shape `(B, L_seq, num_layers, hidden_size_per_layer_input)`.
    fn compute_per_layer_inputs(
        &self,
        input_ids: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;
        let ple_dim = self.hidden_size_per_layer_input;
        let n_layers = self.num_hidden_layers;

        // Token-identity component: scaled embedding lookup, reshaped per-layer.
        let token_id = self.embed_tokens_per_layer.forward(input_ids)?;
        let token_id = (token_id * (ple_dim as f64).sqrt())?;
        let token_id = token_id.reshape((b_sz, seq_len, n_layers, ple_dim))?;

        // Context-aware component: project hidden states, normalize per-layer.
        let proj = self.per_layer_model_projection.forward(inputs_embeds)?;
        let proj = (proj * (self.hidden_size as f64).powf(-0.5))?;
        let proj = proj.reshape((b_sz, seq_len, n_layers, ple_dim))?;
        let proj = self.per_layer_projection_norm.forward(&proj)?;

        // Combine the two streams with the canonical 1/sqrt(2) merge weight.
        (proj + token_id)? * 2f64.powf(-0.5)
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        xs * (self.hidden_size as f64).sqrt()
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let xs = self.embed_tokens(input_ids)?;
        let per_layer_inputs = self.compute_per_layer_inputs(input_ids, &xs)?;
        self.forward_inner(&xs, &per_layer_inputs, seqlen_offset, b_size, seq_len)
    }

    /// Forward pass starting from already-embedded inputs (used by the
    /// multimodal wrapper after image/audio embeddings have been mixed in).
    /// `input_ids` is still required for the token-identity PLE stream.
    pub fn forward_embeds(
        &mut self,
        input_ids: &Tensor,
        xs: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let per_layer_inputs = self.compute_per_layer_inputs(input_ids, xs)?;
        self.forward_inner(xs, &per_layer_inputs, seqlen_offset, b_size, seq_len)
    }

    fn forward_inner(
        &mut self,
        xs: &Tensor,
        per_layer_inputs: &Tensor,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(batch_size, seq_len, seqlen_offset)?;

        // Shared K/V state passed between layers within a single forward pass.
        // The donor layer (last non-shared of each layer_type) writes its full
        // post-RoPE cache here; subsequent shared layers read from it.
        let mut shared_kv = SharedKvStates::default();
        let mut xs = xs.clone();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let per_layer_input = per_layer_inputs.narrow(2, i, 1)?.squeeze(2)?;
            xs = layer.forward(
                &xs,
                &per_layer_input,
                attention_mask.as_ref(),
                sliding_attention_mask.as_ref(),
                seqlen_offset,
                &mut shared_kv,
            )?
        }
        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)?;
        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => Ok(((logits / sc)?.tanh()? * sc)?),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}
