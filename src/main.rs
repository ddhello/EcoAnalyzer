#![recursion_limit = "256"]

use core::panic;
use std::env;

use anyhow::{Ok, Result};
use burn::{ prelude::Backend};
use burn_ndarray::NdArray;
use burn_wgpu::Wgpu;

use crate::{infer::infer, trian::run_training};

mod infer;
mod dataset;
mod tensor;
mod batcher;
mod trian;
mod model;

type MyBackend = Wgpu;
fn main() -> Result<()>{
    let device = <MyBackend as Backend>::Device::default();

    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("train");

    println!("Using device: {:?}", device);

    match mode{
        "train" => {
            run_training(device)?;
        }
        "infer" => {
            let text_to_analyze = args.get(2).expect("Please provide a text to infer");
            println!("Mode: Infer");
            if let Err(e) = infer(device, &text_to_analyze){
                eprintln!("An error occurred during inference: {}", e);
            }
        }
        _=>{
            panic!("Wrong Command!");
        }
    }
    Ok(())
}
