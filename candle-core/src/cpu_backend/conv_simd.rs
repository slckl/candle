use std::borrow::Cow;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{cpu_backend::Map2, Layout, Result, WithDType};

// Import SIMD types for vectorization
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(super) struct Conv2D<'a>(pub(super) &'a crate::conv::ParamsConv2D);

impl Map2 for Conv2D<'_> {
    const OP: &'static str = "conv2d";
    fn f<T: WithDType>(&self, inp: &[T], inp_l: &Layout, k: &[T], k_l: &Layout) -> Result<Vec<T>> {
        let p = self.0;
        let inp = &inp[inp_l.start_offset()..];
        let (inp_s0, inp_s1, inp_s2, inp_s3) = crate::shape::dims4(inp_l.stride())?;
        let k = &k[k_l.start_offset()..];
        let (k_s0, k_s1, k_s2, k_s3) = crate::shape::dims4(k_l.stride())?;
        let (out_h, out_w) = (p.out_h(), p.out_w());

        // Output shape: [b_size, c_out, out_h, out_w].
        let dst = vec![T::zero(); p.b_size * p.c_out * out_h * out_w];

        let start = std::time::Instant::now();
        let cont_s0 = p.i_h * p.i_w * p.c_in;
        let cont_s1 = p.i_w * p.c_in;
        let cont_s2 = p.c_in;
        let layout_is_valid = inp_l.stride() == [cont_s0, cont_s1, cont_s2, 1];
        let inp_cont: Cow<[T]> = if layout_is_valid {
            Cow::Borrowed(inp)
        } else {
            let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.i_h * p.i_w];
            for b_idx in 0..p.b_size {
                for h_idx in 0..p.i_h {
                    for w_idx in 0..p.i_w {
                        for c_idx in 0..p.c_in {
                            let src_idx =
                                b_idx * inp_s0 + c_idx * inp_s1 + h_idx * inp_s2 + w_idx * inp_s3;
                            let dst_idx =
                                b_idx * cont_s0 + h_idx * cont_s1 + w_idx * cont_s2 + c_idx;
                            inp_cont[dst_idx] = inp[src_idx]
                        }
                    }
                }
            }
            Cow::Owned(inp_cont)
        };
        println!("- conv2d copy: {:?}", start.elapsed());

        let start = std::time::Instant::now();
        let k_cache: Vec<Vec<T>> = (0..p.c_out)
            .map(|dst_c_idx| {
                (0..p.k_h * p.k_w)
                    .flat_map(|kw_kh| {
                        let offset_h = kw_kh / p.k_w;
                        let offset_w = kw_kh % p.k_w;
                        (0..p.c_in).map(move |c_in_idx| {
                            k[dst_c_idx * k_s0
                                + c_in_idx * k_s1
                                + offset_h * k_s2
                                + offset_w * k_s3]
                        })
                    })
                    .collect()
            })
            .collect();
        println!("- conv2d k_cache: {:?}", start.elapsed());

        for offset_h in 0..p.k_h {
            let start = std::time::Instant::now();
            for offset_w in 0..p.k_w {
                let start = std::time::Instant::now();
                let k_offset = offset_h * p.k_w + offset_w;

                (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                    let k_cont = &k_cache[dst_c_idx][k_offset * p.c_in..(k_offset + 1) * p.c_in];
                    let base_dst_idx = dst_c_idx * out_w * out_h;

                    for b_idx in 0..p.b_size {
                        let batch_dst_idx = base_dst_idx + b_idx * p.c_out * out_h * out_w;
                        let batch_src_idx = b_idx * cont_s0;

                        for dst_h in 0..out_h {
                            let src_h = p.stride * dst_h + offset_h * p.dilation;
                            if src_h < p.padding || src_h >= p.i_h + p.padding {
                                continue;
                            }
                            let src_h = src_h - p.padding;
                            let h_dst_idx = batch_dst_idx + dst_h * out_w;
                            let h_src_idx = batch_src_idx + src_h * cont_s1;

                            // SIMD-optimized inner loop: compute multiple dot products in parallel
                            simd_conv_inner::<T>(
                                &inp_cont, k_cont, &dst, h_dst_idx, h_src_idx, cont_s2, out_w,
                                p.stride, offset_w, p.dilation, p.padding, p.i_w, p.c_in,
                            );
                        }
                    }
                });
                println!(
                    "--- {offset_h}:{offset_w} conv2d compute: {:?}",
                    start.elapsed()
                );
            }
            println!("-- {offset_h} conv2d compute: {:?}", start.elapsed());
        }

        Ok(dst)
    }
}

// Generic SIMD convolution inner loop dispatcher
#[inline]
fn simd_conv_inner<T: WithDType>(
    inp_cont: &[T],
    k_cont: &[T],
    dst: &[T],
    h_dst_idx: usize,
    h_src_idx: usize,
    cont_s2: usize,
    out_w: usize,
    stride: usize,
    offset_w: usize,
    dilation: usize,
    padding: usize,
    i_w: usize,
    c_in: usize,
) {
    // Dispatch to type-specific SIMD implementations
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        unsafe {
            simd_conv_inner_f32(
                std::mem::transmute(inp_cont),
                std::mem::transmute(k_cont),
                std::mem::transmute(dst),
                h_dst_idx,
                h_src_idx,
                cont_s2,
                out_w,
                stride,
                offset_w,
                dilation,
                padding,
                i_w,
                c_in,
            );
        }
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        unsafe {
            simd_conv_inner_f64(
                std::mem::transmute(inp_cont),
                std::mem::transmute(k_cont),
                std::mem::transmute(dst),
                h_dst_idx,
                h_src_idx,
                cont_s2,
                out_w,
                stride,
                offset_w,
                dilation,
                padding,
                i_w,
                c_in,
            );
        }
    } else {
        // Fallback for other types
        simd_conv_inner_generic::<T>(
            inp_cont, k_cont, dst, h_dst_idx, h_src_idx, cont_s2, out_w, stride, offset_w,
            dilation, padding, i_w, c_in,
        );
    }
}

// SIMD-optimized convolution inner loop for f32
#[inline]
#[cfg(target_arch = "x86_64")]
fn simd_conv_inner_f32(
    inp_cont: &[f32],
    k_cont: &[f32],
    dst: &[f32],
    h_dst_idx: usize,
    h_src_idx: usize,
    cont_s2: usize,
    out_w: usize,
    stride: usize,
    offset_w: usize,
    dilation: usize,
    padding: usize,
    i_w: usize,
    c_in: usize,
) {
    const LANES: usize = 8; // Process 8 output positions in parallel using AVX
    let num_chunks = out_w / LANES;

    unsafe {
        if is_x86_feature_detected!("avx") {
            for chunk in 0..num_chunks {
                let base_w = chunk * LANES;
                let mut accumulators = [_mm256_setzero_ps(); LANES];

                // Compute which output positions are valid
                let mut valid_mask = [true; LANES];
                let mut src_offsets = [0usize; LANES];

                for lane in 0..LANES {
                    let dst_w = base_w + lane;
                    let src_w = stride * dst_w + offset_w * dilation;
                    if src_w < padding || src_w >= i_w + padding {
                        valid_mask[lane] = false;
                    } else {
                        src_offsets[lane] = h_src_idx + (src_w - padding) * cont_s2;
                    }
                }

                // Process channel dimension with SIMD for all valid lanes
                let c_chunks = c_in / 8;

                // Main SIMD loop: process 8 channels at a time for each output position
                for c_chunk in 0..c_chunks {
                    let k_offset = c_chunk * 8;
                    let k_vec = _mm256_loadu_ps(k_cont.as_ptr().add(k_offset));

                    for lane in 0..LANES {
                        if valid_mask[lane] {
                            let inp_ptr = inp_cont.as_ptr().add(src_offsets[lane] + k_offset);
                            let inp_vec = _mm256_loadu_ps(inp_ptr);
                            let prod = _mm256_mul_ps(inp_vec, k_vec);
                            accumulators[lane] = _mm256_add_ps(accumulators[lane], prod);
                        }
                    }
                }

                // Handle remainder channels (scalar fallback)
                let mut remainder_sums = [0.0f32; LANES];
                for c_idx in (c_chunks * 8)..c_in {
                    let k_val = *k_cont.get_unchecked(c_idx);
                    for lane in 0..LANES {
                        if valid_mask[lane] {
                            let inp_val = *inp_cont.get_unchecked(src_offsets[lane] + c_idx);
                            remainder_sums[lane] += inp_val * k_val;
                        }
                    }
                }

                // Horizontal sum and write results
                for lane in 0..LANES {
                    if valid_mask[lane] {
                        let sum = horizontal_sum_avx(accumulators[lane]) + remainder_sums[lane];
                        let dst_idx = h_dst_idx + base_w + lane;
                        let ptr = dst.as_ptr().add(dst_idx) as *mut f32;
                        *ptr += sum;
                    }
                }
            }
        } else {
            // Fallback to SSE or scalar
            simd_conv_inner_generic(
                inp_cont, k_cont, dst, h_dst_idx, h_src_idx, cont_s2, out_w, stride, offset_w,
                dilation, padding, i_w, c_in,
            );
            return;
        }
    }

    // Handle remainder output positions
    for dst_w in (num_chunks * LANES)..out_w {
        let src_w = stride * dst_w + offset_w * dilation;
        if src_w >= padding && src_w < i_w + padding {
            let src_w = src_w - padding;
            let dst_idx = h_dst_idx + dst_w;
            let inp_slice = &inp_cont[h_src_idx + src_w * cont_s2..];

            let mut d = 0.0f32;
            unsafe {
                crate::cpu::vec_dot_f32(inp_slice.as_ptr(), k_cont.as_ptr(), &mut d, c_in);
                let ptr = dst.as_ptr().add(dst_idx) as *mut f32;
                *ptr += d;
            }
        }
    }
}

// AVX horizontal sum helper
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum_avx(v: __m256) -> f32 {
    let sum1 = _mm256_hadd_ps(v, v);
    let sum2 = _mm256_hadd_ps(sum1, sum1);
    let lo = _mm256_castps256_ps128(sum2);
    let hi = _mm256_extractf128_ps(sum2, 1);
    let sum3 = _mm_add_ps(lo, hi);
    _mm_cvtss_f32(sum3)
}

// Fallback for non-x86_64 architectures
#[cfg(not(target_arch = "x86_64"))]
fn simd_conv_inner_f32(
    inp_cont: &[f32],
    k_cont: &[f32],
    dst: &[f32],
    h_dst_idx: usize,
    h_src_idx: usize,
    cont_s2: usize,
    out_w: usize,
    stride: usize,
    offset_w: usize,
    dilation: usize,
    padding: usize,
    i_w: usize,
    c_in: usize,
) {
    simd_conv_inner_generic(
        inp_cont, k_cont, dst, h_dst_idx, h_src_idx, cont_s2, out_w, stride, offset_w, dilation,
        padding, i_w, c_in,
    );
}

// SIMD-optimized convolution inner loop for f64
#[inline]
#[cfg(target_arch = "x86_64")]
fn simd_conv_inner_f64(
    inp_cont: &[f64],
    k_cont: &[f64],
    dst: &[f64],
    h_dst_idx: usize,
    h_src_idx: usize,
    cont_s2: usize,
    out_w: usize,
    stride: usize,
    offset_w: usize,
    dilation: usize,
    padding: usize,
    i_w: usize,
    c_in: usize,
) {
    const LANES: usize = 4; // Process 4 output positions in parallel using AVX
    let num_chunks = out_w / LANES;

    unsafe {
        if is_x86_feature_detected!("avx") {
            for chunk in 0..num_chunks {
                let base_w = chunk * LANES;
                let mut accumulators = [_mm256_setzero_pd(); LANES];

                let mut valid_mask = [true; LANES];
                let mut src_offsets = [0usize; LANES];

                for lane in 0..LANES {
                    let dst_w = base_w + lane;
                    let src_w = stride * dst_w + offset_w * dilation;
                    if src_w < padding || src_w >= i_w + padding {
                        valid_mask[lane] = false;
                    } else {
                        src_offsets[lane] = h_src_idx + (src_w - padding) * cont_s2;
                    }
                }

                let c_chunks = c_in / 4;

                for c_chunk in 0..c_chunks {
                    let k_offset = c_chunk * 4;
                    let k_vec = _mm256_loadu_pd(k_cont.as_ptr().add(k_offset));

                    for lane in 0..LANES {
                        if valid_mask[lane] {
                            let inp_ptr = inp_cont.as_ptr().add(src_offsets[lane] + k_offset);
                            let inp_vec = _mm256_loadu_pd(inp_ptr);
                            let prod = _mm256_mul_pd(inp_vec, k_vec);
                            accumulators[lane] = _mm256_add_pd(accumulators[lane], prod);
                        }
                    }
                }

                let mut remainder_sums = [0.0f64; LANES];
                for c_idx in (c_chunks * 4)..c_in {
                    let k_val = *k_cont.get_unchecked(c_idx);
                    for lane in 0..LANES {
                        if valid_mask[lane] {
                            let inp_val = *inp_cont.get_unchecked(src_offsets[lane] + c_idx);
                            remainder_sums[lane] += inp_val * k_val;
                        }
                    }
                }

                for lane in 0..LANES {
                    if valid_mask[lane] {
                        let sum = horizontal_sum_avx_pd(accumulators[lane]) + remainder_sums[lane];
                        let dst_idx = h_dst_idx + base_w + lane;
                        let ptr = dst.as_ptr().add(dst_idx) as *mut f64;
                        *ptr += sum;
                    }
                }
            }
        } else {
            simd_conv_inner_generic(
                inp_cont, k_cont, dst, h_dst_idx, h_src_idx, cont_s2, out_w, stride, offset_w,
                dilation, padding, i_w, c_in,
            );
            return;
        }
    }

    for dst_w in (num_chunks * LANES)..out_w {
        let src_w = stride * dst_w + offset_w * dilation;
        if src_w >= padding && src_w < i_w + padding {
            let src_w = src_w - padding;
            let dst_idx = h_dst_idx + dst_w;
            let inp_slice = &inp_cont[h_src_idx + src_w * cont_s2..];

            let mut d = 0.0f64;
            // Scalar fallback for f64 (no vec_dot_f64 available in crate::cpu)
            for i in 0..c_in {
                d += inp_slice[i] * k_cont[i];
            }
            unsafe {
                let ptr = dst.as_ptr().add(dst_idx) as *mut f64;
                *ptr += d;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum_avx_pd(v: __m256d) -> f64 {
    let sum1 = _mm256_hadd_pd(v, v);
    let lo = _mm256_castpd256_pd128(sum1);
    let hi = _mm256_extractf128_pd(sum1, 1);
    let sum2 = _mm_add_pd(lo, hi);
    _mm_cvtsd_f64(sum2)
}

#[cfg(not(target_arch = "x86_64"))]
fn simd_conv_inner_f64(
    inp_cont: &[f64],
    k_cont: &[f64],
    dst: &[f64],
    h_dst_idx: usize,
    h_src_idx: usize,
    cont_s2: usize,
    out_w: usize,
    stride: usize,
    offset_w: usize,
    dilation: usize,
    padding: usize,
    i_w: usize,
    c_in: usize,
) {
    simd_conv_inner_generic(
        inp_cont, k_cont, dst, h_dst_idx, h_src_idx, cont_s2, out_w, stride, offset_w, dilation,
        padding, i_w, c_in,
    );
}

// Generic fallback implementation
#[inline]
fn simd_conv_inner_generic<T: WithDType>(
    inp_cont: &[T],
    k_cont: &[T],
    dst: &[T],
    h_dst_idx: usize,
    h_src_idx: usize,
    cont_s2: usize,
    out_w: usize,
    stride: usize,
    offset_w: usize,
    dilation: usize,
    padding: usize,
    i_w: usize,
    c_in: usize,
) {
    for dst_w in 0..out_w {
        let src_w = stride * dst_w + offset_w * dilation;
        if src_w < padding || src_w >= i_w + padding {
            continue;
        }
        let src_w = src_w - padding;
        let dst_idx = h_dst_idx + dst_w;
        let inp_slice = &inp_cont[h_src_idx + src_w * cont_s2..];

        let mut d = T::zero();
        unsafe {
            T::vec_dot(inp_slice.as_ptr(), k_cont.as_ptr(), &mut d, c_in);
            let ptr = dst.as_ptr().add(dst_idx) as *mut T;
            *ptr += d;
        }
    }
}
