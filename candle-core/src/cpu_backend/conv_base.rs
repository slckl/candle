use std::borrow::Cow;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{cpu_backend::Map2, Layout, Result, WithDType};

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
        let inp_cont_len = inp_cont.len();
        // println!("- conv2d copy: {:?}", start.elapsed());
        // println!("- inp_cont.len(): {}", inp_cont.len());

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

        // println!("k_cache.len(): {}", k_cache.len());
        // &p = ParamsConv2D {
        //     b_size: 2,
        //     i_h: 320,
        //     i_w: 320,
        //     k_h: 3,
        //     k_w: 3,
        //     c_out: 16,
        //     c_in: 3,
        //     padding: 1,
        //     stride: 1,
        //     dilation: 1,
        //     cudnn_fwd_algo: None,
        // }

        // let start = std::time::Instant::now();
        for b_idx in 0..p.b_size {
            for offset_h in 0..p.k_h {
                // let start = std::time::Instant::now();
                for offset_w in 0..p.k_w {
                    // let start = std::time::Instant::now();
                    let k_offset = offset_h * p.k_w + offset_w;

                    (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                        // (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                        let k_cont =
                            &k_cache[dst_c_idx][k_offset * p.c_in..(k_offset + 1) * p.c_in];
                        let base_dst_idx = dst_c_idx * out_w * out_h;
                        let batch_dst_idx = base_dst_idx + b_idx * p.c_out * out_h * out_w;
                        let batch_src_idx = b_idx * cont_s0;

                        // out_h = 320
                        for dst_h in 0..out_h {
                            // let start = std::time::Instant::now();
                            let src_h = p.stride * dst_h + offset_h * p.dilation;
                            if src_h < p.padding || src_h >= p.i_h + p.padding {
                                continue;
                            }
                            let src_h = src_h - p.padding;
                            let h_dst_idx = batch_dst_idx + dst_h * out_w;
                            let h_src_idx = batch_src_idx + src_h * cont_s1;

                            // out_w = 320
                            for dst_w in 0..out_w {
                                // let start = std::time::Instant::now();
                                let src_w = p.stride * dst_w + offset_w * p.dilation;
                                if src_w < p.padding || src_w >= p.i_w + p.padding {
                                    continue;
                                }
                                let src_w = src_w - p.padding;
                                let dst_idx = h_dst_idx + dst_w;
                                // println!("dst_idx: {dst_idx}");
                                let inp_idx_1 = h_src_idx + src_w * cont_s2;
                                let inp_idx_2 = (inp_idx_1 + p.c_in).max(inp_cont_len);
                                let inp_cont = &inp_cont[inp_idx_1..inp_idx_2];
                                // let inp_cont = &inp_cont[h_src_idx + src_w * cont_s2..];
                                // println!(
                                //     "inp_cont.len(): {} * k_cont.len(): {}",
                                //     inp_cont.len(),
                                //     k_cont.len(),
                                // );
                                // println!("---- inner loop prep: {:?}", start.elapsed());

                                // let start = std::time::Instant::now();
                                let mut d = T::zero();
                                unsafe {
                                    T::vec_dot(inp_cont.as_ptr(), k_cont.as_ptr(), &mut d, p.c_in)
                                }

                                unsafe {
                                    let ptr = dst.as_ptr().add(dst_idx) as *mut T;
                                    *ptr += d;
                                }
                                // println!("---- inner loop dot: {:?}", start.elapsed());
                            }
                            // ~ 2.1 microseconds
                            // println!(
                            //     "--- {offset_h}:{offset_w}:{dst_h} conv2d compute: {:?}",
                            //     start.elapsed()
                            // );
                        }
                    });
                    // ~ 3 ms
                    // println!(
                    //     "--- {offset_h}:{offset_w} conv2d compute: {:?}",
                    //     start.elapsed()
                    // );
                }
                // println!("-- {offset_h} conv2d compute: {:?}", start.elapsed());
            }
        }
        // println!("- conv2d total compute: {:?}", start.elapsed());

        Ok(dst)
    }
}
