use std::borrow::Cow;

use crate::{conv::ParamsConv2D, cpu_backend::Map2, Layout, Result, WithDType};

// Import SIMD types for vectorization
// #[cfg(target_arch = "aarch64")]
// use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_conv2d_f32<T: WithDType>(
    mut dst: Vec<T>,
    inp_cont: &[T],
    k_cache: &[Vec<T>],
    p: &ParamsConv2D,
    out_h: usize,
    out_w: usize,
) -> Vec<T> {
    // Transmute to f32 slices for direct SIMD operations
    let inp_f32: &[f32] = std::mem::transmute(inp_cont);
    let dst_f32: &[f32] = std::mem::transmute(dst.as_mut_slice());

    let stride_h = p.stride;
    let stride_w = p.stride;
    let dilation = p.dilation;
    let padding = p.padding;

    let inp_h = p.i_h;
    let inp_w = p.i_w;
    let c_in = p.c_in;
    let k_h = p.k_h;
    let k_w = p.k_w;

    // Process each batch
    for b_idx in 0..p.b_size {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        let inp_batch_offset = b_idx * inp_h * inp_w * c_in;
        let dst_batch_offset = b_idx * p.c_out * out_h * out_w;

        // Process each output channel
        (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
            let k_data_f32: &[f32] = std::mem::transmute(k_cache[dst_c_idx].as_slice());
            let dst_channel_offset = dst_batch_offset + dst_c_idx * out_h * out_w;

            // Process each output position
            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    let dst_idx = dst_channel_offset + out_y * out_w + out_x;

                    // Calculate input position
                    let in_y_base = out_y * stride_h;
                    let in_x_base = out_x * stride_w;

                    // Accumulator for this output position
                    let mut acc = _mm256_setzero_ps();

                    let mut k_idx = 0;

                    // Iterate over kernel window
                    for ky in 0..k_h {
                        for kx in 0..k_w {
                            let in_y = in_y_base + ky * dilation;
                            let in_x = in_x_base + kx * dilation;

                            // Check bounds with padding
                            if in_y >= padding && in_x >= padding {
                                let in_y_actual = in_y - padding;
                                let in_x_actual = in_x - padding;

                                if in_y_actual < inp_h && in_x_actual < inp_w {
                                    let inp_offset = inp_batch_offset
                                        + in_y_actual * inp_w * c_in
                                        + in_x_actual * c_in;

                                    // Process input channels in chunks of 8
                                    let mut c_idx = 0;
                                    while c_idx + 8 <= c_in {
                                        let inp_vec = _mm256_loadu_ps(&inp_f32[inp_offset + c_idx]);
                                        let k_vec = _mm256_loadu_ps(&k_data_f32[k_idx + c_idx]);
                                        acc = _mm256_fmadd_ps(inp_vec, k_vec, acc);
                                        c_idx += 8;
                                    }

                                    // Handle remaining channels
                                    while c_idx < c_in {
                                        let ptr = dst_f32.as_ptr().add(dst_idx) as *mut f32;
                                        *ptr +=
                                            inp_f32[inp_offset + c_idx] * k_data_f32[k_idx + c_idx];
                                        // dst_f32[dst_idx] +=
                                        //     inp_f32[inp_offset + c_idx] * k_data_f32[k_idx + c_idx];
                                        c_idx += 1;
                                    }
                                }
                            }

                            k_idx += c_in;
                        }
                    }

                    // Horizontal sum of accumulator
                    // acc = [a0, a1, a2, a3, a4, a5, a6, a7]
                    let acc_high = _mm256_extractf128_ps(acc, 1); // [a4, a5, a6, a7]
                    let acc_low = _mm256_castps256_ps128(acc); // [a0, a1, a2, a3]
                    let acc_sum = _mm_add_ps(acc_low, acc_high); // [a0+a4, a1+a5, a2+a6, a3+a7]

                    let acc_sum_high = _mm_movehl_ps(acc_sum, acc_sum); // [a2+a6, a3+a7, ?, ?]
                    let acc_sum2 = _mm_add_ps(acc_sum, acc_sum_high); // [a0+a4+a2+a6, a1+a5+a3+a7, ?, ?]

                    let acc_sum_high2 = _mm_shuffle_ps(acc_sum2, acc_sum2, 0x1);
                    let acc_final = _mm_add_ss(acc_sum2, acc_sum_high2);

                    // dst_f32[dst_idx] += _mm_cvtss_f32(acc_final);
                    let sum = _mm_cvtss_f32(acc_final);
                    let ptr = dst_f32.as_ptr().add(dst_idx) as *mut f32;
                    *ptr += sum;
                }
            }
        });
    }

    dst
}

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

        // let start = std::time::Instant::now();
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
        // println!("- conv2d copy: {:?}", start.elapsed());

        // let start = std::time::Instant::now();
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
        // println!("- conv2d k_cache: {:?}", start.elapsed());

        // AVX2 optimized version for f32
        // let start = std::time::Instant::now();

        // Check if we're working with f32, panic otherwise
        if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>() {
            panic!("AVX2 SIMD convolution only supports f32 dtype");
        }

        // SAFETY: We've verified T is f32 above
        let dst = unsafe { avx2_conv2d_f32(dst, &inp_cont, &k_cache, p, out_h, out_w) };

        // println!("- conv2d compute: {:?}", start.elapsed());

        Ok(dst)
    }
}
