use core::panic;
use std::sync::Arc;

use anyhow::Result;
use burn::backend::Autodiff;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn::record::{CompactRecorder, DefaultRecorder};
use burn::train::metric::LossMetric;
use burn::train::{LearnerBuilder, TrainStep};
use burn_ndarray::NdArray;
use burn_wgpu::Wgpu;
use crate::batcher::ClassificationBatch;
use crate::dataset::MyDataset;
use crate::model::TextRegressionModel;
use crate::{
    batcher::ClassificationBatcher,
    dataset::{ProcessedItem, TextClassificationBatcher},
    tensor::Vocab,
};


pub type MyBackend = Wgpu;
pub type MyAutodiffBackend = Autodiff<MyBackend>;

pub fn run_training(device: <MyBackend as Backend>::Device) -> Result<()> {
    //读取训练数据
    let tokenizer_batcher = TextClassificationBatcher::read_from_csv("src/data.csv".to_string())?;
    //分词
    let tokenzied_data = tokenizer_batcher.tokenize();
    //创建词典
    let vocab = Vocab::build(&tokenzied_data, 1);
    //预处理序列
    let processed_raw = tokenzied_data.to_processed_sequences(&vocab);

    // 将 (Vec<u32>, u32) 转换为 ProcessedItem
    let processed_items: Vec<ProcessedItem> = processed_raw
        .into_iter()
        .map(|(ids, score)| ProcessedItem {
            token_ids: ids,
            score,
        })
        .collect();

    let total_size = processed_items.len();
    if total_size < 10{
        panic!("Dataset too small!");
    }
    let train_size = (total_size as f32 * 0.8) as usize;

    let train_items = processed_items[..train_size].to_vec();
    let valid_items = processed_items[train_size..].to_vec();

    println!("Dataset split: {} training items, {} validation items.", train_items.len(), valid_items.len());
    
    let batcher_train = ClassificationBatcher;
    let batcher_valid = ClassificationBatcher;

    let dataset_train = MyDataset::new(train_items);
    let dataset_valid = MyDataset::new(valid_items);

    let batch_size = 8;

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .num_workers(4)
        .build(dataset_train);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(4)
        .build(dataset_valid);

    println!("Dataloader Created.");

    //开始模型训练
    const D_MODEL: usize = 128;
    const N_HEADS: usize = 4;
    const NUM_LAYERS: usize = 4;
    const DROPOUT: f64 = 0.1;

    let model = TextRegressionModel::<MyAutodiffBackend>::new(
        vocab.len(),
        D_MODEL,
        N_HEADS,
        NUM_LAYERS,
        D_MODEL*4,
        DROPOUT,
        &device,
    );

    let optim = AdamConfig::new().init();

    //配置learner
    println!("Setting up learner");
    let learner = LearnerBuilder::new("articats")
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(10)
        .build(model, optim, 1e-4);

    println!("Starting Training....");
    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    println!("Training Completed!");

    model_trained.save_file("final_model", &DefaultRecorder::new())?;

    Ok(())
}
