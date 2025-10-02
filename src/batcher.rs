use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{Int, TensorData};
use burn::{prelude::Backend, tensor::Tensor};

use crate::{dataset::ProcessedItem};

#[derive(Debug,Clone)]
pub struct ClassificationBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,  //[批次大小,段长度]
    pub targets: Tensor<B, 2>, //[批次大小] 
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
        
        let all_tokens_data: Vec<Vec<u32>> = items.iter().map(|item| item.token_ids.clone()).collect();
        let all_scores_data: Vec<f32> = items.iter().map(|item| item.score as f32).collect();

        let token_flat: Vec<i32> = all_tokens_data.into_iter().flatten().map(|t| t as i32).collect();
        let seq_len = token_flat.len() / batch_size;

        let tokens_data = TensorData::new(token_flat,[batch_size,seq_len]);
        let tokens = Tensor::<B,2,Int>::from_data(tokens_data.convert::<B::IntElem>(), device);

        let target_data = TensorData::new(all_scores_data,[batch_size,1]);
        let targets = Tensor::<B,2>::from_data(target_data, device);

        ClassificationBatch { tokens, targets }
    }
}
