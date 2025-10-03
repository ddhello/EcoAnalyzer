use std::char::MAX;

use anyhow::Result;
use burn::optim::record;
use burn::tensor::ops::IntElem;
use burn::{backend, prelude::*};
use burn::record::{CompactRecorder, Recorder};
use burn_wgpu::IntElement;

use crate::dataset::TextClassificationBatcher;
use crate::model::TextRegressionModel;
use crate::tensor::{Vocab, PAD_ID};
use crate::trian::{MyAutodiffBackend, MyBackend};

pub fn infer(device: <MyBackend as Backend>::Device,text_to_infer: &str) -> Result<()>{
    println!("loading tokenizer and vocab...");

    let tokenizer_batcher = TextClassificationBatcher::read_from_csv("src/data.csv".to_string())?;
    let tokenizer_data = tokenizer_batcher.tokenize();
    let vocab = Vocab::build(&tokenizer_data, 1);
    let vocab_size = vocab.len();

    const D_MODEL: usize = 256;
    const N_HEADS: usize = 4;
    const NUM_LAYERS: usize = 6;
    const DROPOUT: f64 = 0.25;
    const MAX_SEQ_LEN: usize = 128;

    println!("Loading Trained Model....");

    let model = TextRegressionModel::<MyBackend>::new(
        vocab_size,
        D_MODEL,
        N_HEADS,
        NUM_LAYERS,
        D_MODEL*4,
        DROPOUT,
        &device
    );

    let record = CompactRecorder::new()
        .load("final_model".into(), &device)?;

    let model = model.load_record(record);
    
    println!("Model Loaded Successfully!");

    println!("Preprocessing input text....");

    let tokens = TextClassificationBatcher::tokenize_single(text_to_infer);

    let mut token_ids: Vec<u32> = tokens
        .iter()
        .map(|token| vocab.get_id(token))
        .collect();
    
    if token_ids.len() > MAX_SEQ_LEN{
        println!("Text is too long, Truncating...");
        token_ids.truncate(MAX_SEQ_LEN);
    }

    let padding_needed = MAX_SEQ_LEN - token_ids.len();
    if padding_needed > 0 {
        println!("Padding with {} tokens.", padding_needed);
        let pad_id = PAD_ID;
        for _ in 0..padding_needed {
            token_ids.push(pad_id);
        }
    }

    let token_ids_i32: Vec<i32> = token_ids.into_iter().map(|id| id as i32).collect();
    let tokens_data = TensorData::new(token_ids_i32, [1,MAX_SEQ_LEN]);
    
    let token_tensor = Tensor::<MyBackend,2,Int>::from_data(
        tokens_data.convert::<<MyBackend as Backend>::IntElem>(),
        &device,
    );

    println!("Input tensor created with shape: {:?}", token_tensor.shape());

    println!("Running Inference....");

    let output_tensor = model.forward(token_tensor);

    let score = output_tensor.into_scalar();

    println!("\n========================================");
    println!("待分析文本: \"{}\"", text_to_infer);
    println!("模型预测乐观度: {:.2}", score.max(1.0).min(5.0));
    println!("========================================");

    Ok(())
}