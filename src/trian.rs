use std::sync::Arc;

use anyhow::Result;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn_ndarray::NdArray;
use crate::batcher::ClassificationBatch;
use crate::dataset::MyDataset;
use crate::{
    batcher::ClassificationBatcher,
    dataset::{ProcessedItem, TextClassificationBatcher},
    tensor::Vocab,
};

pub fn run_train_setup() -> Result<()> {
    //读取训练数据
    let batcher = TextClassificationBatcher::read_from_csv("src/data.csv".to_string())?;
    //分词
    let tokenzied_data = batcher.tokenize();
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

    //设置burn后端
    type B = NdArray;
    let batcher = ClassificationBatcher;

    //创建dataloader
    let dataset = MyDataset::new(processed_items);

    let batch_size = 2;

    let dataloader: Arc<dyn DataLoader<B, ClassificationBatch<B>>> =
        DataLoaderBuilder::<B, ProcessedItem, ClassificationBatch<B>>::new(batcher)
            .batch_size(batch_size)
            .num_workers(1)
            .build(dataset);

    println!("Created DataLoader with DataLoaderBuilder. Iterating through batches...");

    for (batch_idx, batch_tensor) in dataloader.iter().enumerate().take(2) {
        println!("\n--- 批次 {} ---", batch_idx);
        println!("Tokens Tensor 形状: {:?}", batch_tensor.tokens.shape());
        println!("Targets Tensor 形状: {:?}", batch_tensor.targets.shape());
        println!(
            "Tokens 数据 (片段):\n{}",
            batch_tensor.tokens.clone().slice([0..1, 0..5])
        );
        println!(
            "Targets 数据 (片段):\n{}",
            batch_tensor.targets.clone().slice([0..1])
        );
    }

    Ok(())
}
