use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig};
use criterion::{black_box, criterion_group, Criterion};
use std::time::Instant;

const B: usize = 1;
const C: usize = 1;
const M: usize = 128;
const K: usize = 128;

fn run(input: Tensor, weight: Tensor, bias: Option<Tensor>, config: Conv2dConfig) {
    Conv2d::new(weight, bias, config).forward(&input).unwrap();
}

fn run_conv2d_benchmark(
    c: &mut Criterion,
    device: &Device,
    dtype: DType,
    k_size: usize,
    bias: bool,
    name: &str,
) {
    let weight = Tensor::ones((1, 1, k_size, k_size), dtype, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bias = if bias {
        Some(Tensor::zeros(M, dtype, device).unwrap())
    } else {
        None
    };
    let input = Tensor::ones((B, C, M, K), dtype, device).unwrap();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(
                    black_box(input.clone()),
                    black_box(weight.clone()),
                    black_box(bias.clone()),
                    Default::default(),
                );
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = BenchDeviceHandler::new().unwrap();
    for d in device.devices {
        // run_conv2d_benchmark(c, &d, DType::F32, 1, true, "conv2d_f32_1x1");
        // run_conv2d_benchmark(c, &d, DType::F32, 3, true, "conv2d_f32_3x3");
        // run_conv2d_benchmark(c, &d, DType::F32, 5, true, "conv2d_f32_5x5");
        run_conv2d_benchmark(c, &d, DType::F32, 3, false, "conv2d_f32_3x3_no_bias");
        run_conv2d_benchmark(c, &d, DType::F32, 5, false, "conv2d_f32_5x5_no_bias");
        // run_conv2d_benchmark(c, &d, DType::F16, 1, true, "conv2d_f16_1x1");
        // run_conv2d_benchmark(c, &d, DType::F16, 3, true, "conv2d_f16_3x3");
        // run_conv2d_benchmark(c, &d, DType::F16, 5, true, "conv2d_f16_5x5");
        run_conv2d_benchmark(c, &d, DType::F16, 5, false, "conv2d_f16_5x5_no_bias");
    }
}

criterion_group!(benches, criterion_benchmark);
