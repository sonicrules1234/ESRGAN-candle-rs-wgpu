use candle_core::{pickle, DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use ffmpeg_frame_grabber::{FFMpegVideo, FFMpegVideoOptions};
use std::io::{Cursor, Read, Write};
use std::process::Stdio;
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
mod new_arch;
mod old_arch;
use candle_core::safetensors::load;
use clap::ValueEnum;
use compact::SRVGGNetCompact as Compact;
use image::DynamicImage;
use image::RgbImage;
use new_arch::RRDBNet as RealESRGAN;
use old_arch::RRDBNet as OldESRGAN;
use std::path::Path;
mod old_arch_helpers;
use old_arch_helpers::{get_in_nc, get_nb, get_nf, get_out_nc, get_scale};
mod compact;

use clap::Parser;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ModelType {
    /// Old-arch ESRGAN
    Old,
    /// New-arch ESRGAN (RealESRGAN)
    New,
    // RealESRGANv2 aka Compact
    Compact,
}

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the model file in safetensors format
    #[arg(short, long)]
    model: String,

    /// Path to input video file
    #[arg(short, long)]
    input: String,

    /// Path to output video file
    #[arg(short, long)]
    output: String,

    /// Device to run the model on
    /// -1 for CPU, 0 for GPU 0, 1 for GPU 1, etc.
    #[arg(short, long, default_value = "0")]
    device: i32,

    /// Architecture revision (old or new). Dependent on the model used.
    #[arg(short, long, value_enum)]
    arch: Option<ModelType>,

    /// Number of input channels. Dependent on the model used.
    #[arg(long)]
    in_channels: Option<usize>,

    /// Number of output channels. Dependent on the model used.
    #[arg(long)]
    out_channels: Option<usize>,

    /// Number of RRDB blocks. Dependent on the model used.
    #[arg(long)]
    num_blocks: Option<usize>,

    /// Number of features. Dependent on the model used.
    #[arg(long)]
    num_features: Option<usize>,

    /// Scale of the model. Dependent on the model used.
    #[arg(short, long)]
    scale: Option<usize>,

    /// Run the model with half precision (fp16)
    #[arg(long)]
    half: bool,
}

fn img2tensor(img: DynamicImage, device: &Device, half: bool) -> Tensor {
    let height: usize = img.height() as usize;
    let width: usize = img.width() as usize;
    let data = img.to_rgb8().into_raw();
    let tensor = Tensor::from_vec(data, (height, width, 3), &Device::Cpu)
        .unwrap()
        .permute((2, 0, 1))
        .unwrap();
    let image_t = (tensor
        .unsqueeze(0)
        .unwrap()
        .to_dtype(if half { DType::F16 } else { DType::F32 })
        .unwrap()
        / 255.)
        .unwrap()
        .to_device(device)
        .unwrap();
    return image_t;
}

fn tensor2img(tensor: Tensor) -> RgbImage {
    let cpu = Device::Cpu;

    let result = tensor
        .permute((1, 2, 0))
        .unwrap()
        .detach()
        .to_device(&cpu)
        .unwrap()
        .to_dtype(DType::U8)
        //.unwrap()
        //.to_device(&cpu)
        .unwrap();

    let dims = result.dims();
    let height = dims[0];
    let width = dims[1];

    let data = result.flatten_to(2).unwrap().to_vec1::<u8>().unwrap();
    let out_img = RgbImage::from_vec(width as u32, height as u32, data).unwrap();
    out_img
}

enum ModelVariant {
    Old(OldESRGAN),
    New(RealESRGAN),
    Compact(Compact),
}

fn process(model: &ModelVariant, img: DynamicImage, device: &Device, half: bool) -> RgbImage {
    let img_t = img2tensor(img, &device, half);

    //let now = Instant::now();
    let result = match model {
        ModelVariant::Old(model) => model.forward(&img_t).unwrap(),
        ModelVariant::New(model) => model.forward(&img_t).unwrap(),
        ModelVariant::Compact(model) => model.forward(&img_t).unwrap(),
    };
    //println!("Model took {:?}", now.elapsed());

    let result = (result.squeeze(0).unwrap().clamp(0., 1.).unwrap() * 255.).unwrap();

    let out_img = tensor2img(result);
    return out_img;
}
#[tokio::main]
async fn main() {
    let args: Args = Args::parse();

    let device = match args.device {
        -1 => Device::Cpu,
        _ => {
            let config = candle_core::WgpuDeviceConfig::default();
            Device::new_wgpu_sync_config(0, config).unwrap()
        }
    };

    let path_extension = Path::new(&args.model)
        .extension()
        .unwrap()
        .to_str()
        .unwrap();

    let state_dict = match path_extension {
        "safetensors" => load(&args.model, &device).unwrap(),
        "pth" => pickle::read_all_with_key(&args.model, Some("params_ema"))
            .unwrap()
            .into_iter()
            .collect(),
        _ => panic!("Invalid model file extension"),
    };

    let vb = {
        VarBuilder::from_tensors(
            state_dict.clone(),
            if args.half { DType::F16 } else { DType::F32 },
            &device,
        )
    };

    let model_arch =
        args.arch
            .unwrap_or(if state_dict.keys().any(|x| x.contains("model.0.weight")) {
                ModelType::Old
            } else {
                ModelType::New
            });

    let model: ModelVariant = match model_arch {
        ModelType::Old => ModelVariant::Old(
            OldESRGAN::load(
                vb,
                args.in_channels.unwrap_or(get_in_nc(&state_dict)),
                args.out_channels.unwrap_or(get_out_nc(&state_dict)),
                args.scale.unwrap_or(get_scale(&state_dict)),
                args.num_features.unwrap_or(get_nf(&state_dict)),
                args.num_blocks.unwrap_or(get_nb(&state_dict)),
                32,
            )
            .unwrap(),
        ),
        ModelType::New => ModelVariant::New(
            RealESRGAN::load(
                vb,
                args.in_channels.unwrap_or(3),
                args.out_channels.unwrap_or(3),
                args.scale.unwrap_or(4),
                args.num_features.unwrap_or(64),
                args.num_blocks.unwrap_or(23),
                32,
            )
            .unwrap(),
        ),
        ModelType::Compact => ModelVariant::Compact(
            Compact::load(
                vb,
                args.in_channels.unwrap_or(3),
                args.out_channels.unwrap_or(3),
                args.num_features.unwrap_or(64),
                args.num_blocks.unwrap_or(23),
                args.scale.unwrap_or(4),
            )
            .unwrap(),
        ),
    };

    let video_path = args.input;
    let out_video_path = args.output;

    let now = Instant::now();
    let mut fc_cmd = Command::new("ffprobe");
    let mut fc_args: Vec<&str> =
        "-v error -select_streams v:0 -of csv=p=0 -count_frames -show_entries stream=nb_read_frames"
            .split(" ")
            .collect();
    fc_args.push(&video_path);
    let fc_command_output = fc_cmd.args(fc_args.clone()).output().await.unwrap();
    let mut fc_out_string = String::new();
    fc_command_output
        .stdout
        .as_slice()
        .read_to_string(&mut fc_out_string)
        .unwrap();
    let frame_count: usize = fc_out_string.trim().to_string().parse().unwrap();
    let mut fr_cmd = Command::new("ffprobe");
    let mut fr_args: Vec<&str> =
        "-v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate"
            .split(" ")
            .collect();
    fr_args.push(&video_path);
    let fr_command_output = fr_cmd.args(fr_args.clone()).output().await.unwrap();
    let mut fr_out_string = String::new();
    fr_command_output
        .stdout
        .as_slice()
        .read_to_string(&mut fr_out_string)
        .unwrap();
    let frame_rate = fr_out_string.trim().to_string();
    let mut ffmpeg_cmd = Command::new("ffmpeg");
    let ffmpeg_args_string = format!("-r {frame_rate} -f image2pipe -i pipe: -vn -i {video_path} -c:a copy -c:v libx264rgb -crf 0 -qp 0 -preset veryslow -r {frame_rate} {out_video_path}");
    let mut ffmpeg_spawn = ffmpeg_cmd
        .args(ffmpeg_args_string.split(" ").collect::<Vec<&str>>())
        .kill_on_drop(true)
        .stdin(Stdio::piped())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let mut ffmpeg_stdin = ffmpeg_spawn.stdin.take().unwrap();
    //let stdin_bufwriter = BufWriter::new(ffmpeg_stdin);
    let video = FFMpegVideo::open(Path::new(&video_path), FFMpegVideoOptions::default()).unwrap();
    let mut count = 0;
    for frame in video {
        count += 1;
        if count > frame_count {
            break;
        }
        if let Ok(img) = frame {
            // do something with the image data here ...
            //let img = image::load_from_memory_with_format(&png_img_data, image::ImageFormat::Png)
            //.unwrap();
            let out_img = process(
                &model,
                DynamicImage::ImageRgb8(img.image),
                &device,
                args.half,
            );
            let mut image_vec: Vec<u8> = Vec::new();
            out_img
                .write_to(&mut Cursor::new(&mut image_vec), image::ImageFormat::Png)
                .unwrap();
            ffmpeg_stdin
                .write_all(&mut image_vec.as_slice())
                .await
                .unwrap();
        } else {
            break;
        }
    }
    drop(ffmpeg_stdin);
    ffmpeg_spawn.wait().await.unwrap();
    println!("Time taken: {:?}", now.elapsed());
}
