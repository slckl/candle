use std::time::{Duration, Instant};

use anyhow::Result;
use candle::{DType, Device, Module, Tensor};
use candle_nn::{Conv2dConfig, VarBuilder, VarMap};

fn test_tensor(dims: (usize, usize, usize, usize), device: &Device) -> Result<Tensor> {
    let (batch_size, channels, height, width) = dims;

    // Create deterministic input tensor with hardcoded values for reproducibility
    let mut input_data = vec![0.0f32; batch_size * channels * height * width];

    for b in 0..batch_size {
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    // Create a deterministic pattern based on batch, channel, and position
                    let value = (b + 1) as f32 * 0.1
                        + (c + 1) as f32 * 0.01
                        + (h * width + w) as f32 * 0.001;
                    let idx = b * channels * height * width + c * height * width + h * width + w;
                    input_data[idx] = value;
                }
            }
        }
    }

    let input_tensor = Tensor::from_vec(input_data, (batch_size, channels, height, width), device)?;

    Ok(input_tensor)
}

#[test]
fn just_conv() -> Result<()> {
    // let device = Device::new_cuda(0)?;
    let device = Device::Cpu;

    // Create deterministic input: batch_size=2, channels=3, height=32, width=32
    let (batch_size, in_channels, height, width) = (2, 3, 320, 320);
    let dims = (batch_size, in_channels, height, width);

    let input_tensor = test_tensor(dims, &device)?;

    // Create conv2d layer: 3 input channels, 16 output channels, 3x3 kernel, stride=1, padding=1
    let (out_channels, kernel_size, stride, padding) = (16, 3, 1, 1);

    // Create VarMap and VarBuilder for the conv layer
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let conv_config = Conv2dConfig {
        padding,
        stride,
        dilation: 1,
        groups: 1,
        cudnn_fwd_algo: None,
    };

    let conv_layer = candle_nn::conv2d_no_bias(
        in_channels,
        out_channels,
        kernel_size,
        conv_config,
        vb.pp("conv"),
    )?;
    const ITERS: usize = 1000;
    const WARMUP: usize = 100;
    let mut min = Duration::MAX;
    let mut max = Duration::ZERO;
    let mut total = Duration::ZERO;
    for i in 0..ITERS + WARMUP {
        // println!("---");
        let start = Instant::now();
        let _ = conv_layer.forward(&input_tensor)?;
        device.synchronize()?;
        let elapsed = start.elapsed();
        if i > WARMUP {
            total += elapsed;
            if elapsed < min {
                min = elapsed;
            }
            if elapsed > max {
                max = elapsed;
            }
        }
        // println!("conv2d {:?}", start.elapsed());
    }
    device.synchronize()?;

    println!(
        "{ITERS} iters, min/avg/max: [{:?} {:?} {:?}]",
        min,
        total / ITERS as u32,
        max
    );

    Ok(())
}
