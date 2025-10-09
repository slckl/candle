use std::borrow::Cow;

use crate::{cpu_backend::Map2, Layout, Result, WithDType};

pub(super) struct Conv2D<'a>(pub(super) &'a crate::conv::ParamsConv2D);

impl Map2 for Conv2D<'_> {
    const OP: &'static str = "conv2d";
    fn f<T: WithDType + num_traits::Num + Copy + 'static>(
        &self,
        inp: &[T],
        inp_l: &Layout,
        k: &[T],
        k_l: &Layout,
    ) -> Result<Vec<T>> {
        let p = self.0;
        let inp = &inp[inp_l.start_offset()..];
        let (inp_s0, inp_s1, inp_s2, inp_s3) = crate::shape::dims4(inp_l.stride())?;
        let k = &k[k_l.start_offset()..];
        let (k_s0, k_s1, k_s2, k_s3) = crate::shape::dims4(k_l.stride())?;
        let (out_h, out_w) = (p.out_h(), p.out_w());

        // Output shape: [b_size, c_out, out_h, out_w].
        let mut dst = vec![T::zero(); p.b_size * p.c_out * out_h * out_w];

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

        // let start = std::time::Instant::now();

        // Implicit GEMM: process output in cache-friendly tiles
        // Tile size (number of output pixels to process at once)
        const TILE_SIZE: usize = 128;

        let k_size = p.c_in * p.k_h * p.k_w;
        let total_out_pixels = out_h * out_w;

        // Process each batch
        for b_idx in 0..p.b_size {
            let inp_offset = b_idx * cont_s0;
            let out_batch_offset = b_idx * (p.c_out * out_h * out_w);

            // Process output in tiles
            for tile_start in (0..total_out_pixels).step_by(TILE_SIZE) {
                let tile_end = (tile_start + TILE_SIZE).min(total_out_pixels);
                let tile_size = tile_end - tile_start;

                // Build im2col tile: [k_size, tile_size]
                // This represents the input patches needed for this tile of outputs
                let mut col_tile = vec![T::zero(); k_size * tile_size];

                for tile_idx in 0..tile_size {
                    let out_pixel_idx = tile_start + tile_idx;
                    let out_y = out_pixel_idx / out_w;
                    let out_x = out_pixel_idx % out_w;

                    // Extract the im2col patch for this output position
                    let mut patch_offset = 0;
                    for kh in 0..p.k_h {
                        for kw in 0..p.k_w {
                            let in_y =
                                (out_y * p.stride + kh * p.dilation) as isize - p.padding as isize;
                            let in_x =
                                (out_x * p.stride + kw * p.dilation) as isize - p.padding as isize;

                            if in_y >= 0
                                && in_y < p.i_h as isize
                                && in_x >= 0
                                && in_x < p.i_w as isize
                            {
                                let in_y = in_y as usize;
                                let in_x = in_x as usize;
                                for c_in in 0..p.c_in {
                                    let inp_idx =
                                        inp_offset + in_y * cont_s1 + in_x * cont_s2 + c_in;
                                    let col_idx = patch_offset * tile_size + tile_idx;
                                    col_tile[col_idx] = inp_cont[inp_idx];
                                    patch_offset += 1;
                                }
                            } else {
                                // Padding: already zero
                                patch_offset += p.c_in;
                            }
                        }
                    }
                }

                // Now perform matmul: k_cache [c_out, k_size] @ col_tile [k_size, tile_size]
                // Use MatMul::f for efficient computation
                let matmul = super::MatMul((1, p.c_out, tile_size, k_size));

                // Flatten k_cache into a single vector for matmul
                let k_flat: Vec<T> = k_cache.iter().flat_map(|row| row.iter().copied()).collect();

                // Create layouts for matmul
                // k_flat layout: [c_out, k_size] with stride [k_size, 1]
                let k_layout = Layout::contiguous((p.c_out, k_size));

                // col_tile layout: [k_size, tile_size] with stride [tile_size, 1]
                let col_layout = Layout::contiguous((k_size, tile_size));

                // Perform matmul
                let result = matmul.f(&k_flat, &k_layout, &col_tile, &col_layout)?;

                // Copy results to output: result is [c_out, tile_size]
                for c_out_idx in 0..p.c_out {
                    for tile_idx in 0..tile_size {
                        let out_pixel_idx = tile_start + tile_idx;
                        let out_y = out_pixel_idx / out_w;
                        let out_x = out_pixel_idx % out_w;
                        let dst_idx =
                            out_batch_offset + c_out_idx * (out_h * out_w) + out_y * out_w + out_x;
                        let result_idx = c_out_idx * tile_size + tile_idx;
                        dst[dst_idx] = result[result_idx];
                    }
                }
            }
        }

        // println!("- conv2d compute: {:?}", start.elapsed());

        Ok(dst)
    }
}
