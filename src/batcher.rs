use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{Int, TensorData};
use burn::{prelude::Backend, tensor::Tensor};

use crate::{dataset::ProcessedItem, tensor::MAX_SEQ_LEN};

#[derive(Debug,Clone)]
pub struct ClassificationBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,  //[批次大小,段长度]
    pub targets: Tensor<B, 1, Int>, //[批次大小]
}

pub struct ClassificationBatcher;

// impl<B:Backend> ClassificationBatcher<B> {
//     fn new(device: B::Device) -> Self{
//         Self { device }
//     }
// }

impl<B: Backend> Batcher<B, ProcessedItem, ClassificationBatch<B>> for ClassificationBatcher {
    fn batch(&self, items: Vec<ProcessedItem>, device: &B::Device) -> ClassificationBatch<B> {
        // 1. 收集 tokens 和 targets
        let batch_size = items.len();
        let mut tokens_data = Vec::with_capacity(batch_size * MAX_SEQ_LEN);
        let mut targets_data = Vec::with_capacity(batch_size);

        for item in items {
            // 收集 Token IDs (flattened)
            tokens_data.extend(item.token_ids);
            // 收集 Score (作为标签)
            targets_data.push(item.score);
        }

        let tokens_vec = tokens_data.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let targets_vec = targets_data.iter().map(|&x| x as f32).collect::<Vec<f32>>();

        let tokens = Tensor::<B, 1>::from_floats(
            TensorData::new(tokens_vec, [batch_size * MAX_SEQ_LEN]),
            device,
        )
        .reshape([batch_size, MAX_SEQ_LEN])
        .int();

        let targets =
            Tensor::<B, 1>::from_floats(TensorData::new(targets_vec, [batch_size]), device).int();

        ClassificationBatch { tokens, targets }
    }
}
