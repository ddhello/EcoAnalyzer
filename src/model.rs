use burn::{module::Module, nn::{loss::{MseLoss, Reduction}, transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput}, Embedding, EmbeddingConfig, Linear, LinearConfig}, optim::GradientsParams, prelude::Backend, tensor::{Int, Tensor}, train::{ClassificationOutput, RegressionOutput, TrainOutput, TrainStep, ValidStep}};

use burn::tensor::backend::AutodiffBackend;

use crate::{batcher::ClassificationBatch, model};


#[derive(Module,Debug)]
pub struct TextRegressionModel<B:Backend>{
    //嵌入层： 将token映射到d_model维的向量
    embedding: Embedding<B>,
    //Transformer编码器
    transformer: TransformerEncoder<B>,
    //输出层：线性变换，将transformer的输出映射到具体的分数
    output_layer: Linear<B>,
}

impl<B:Backend> TextRegressionModel<B>{
    pub fn new(
        vocab_size: usize, //字典大小
        d_model: usize, //模型内部维度
        n_heads: usize, //Transformer多头注意力数量
        num_layers: usize, //Transformer编码器堆叠的层数
        d_ff: usize,
        dropout: f64, //丢弃率
        device: &B::Device,
    )->Self{
        let embedding_config = EmbeddingConfig::new(vocab_size, d_model);

        let transformer_config = TransformerEncoderConfig::new(d_model, d_ff, n_heads, num_layers)
            .with_dropout(dropout);

        let output_layer_config = LinearConfig::new(d_model, 1);

        Self{
            embedding: embedding_config.init(device),
            transformer: transformer_config.init(device),
            output_layer: output_layer_config.init(device),
        }
    }

    pub fn forward(&self,tokens: Tensor<B,2,Int>) -> Tensor<B,2>{
        //将tokenID转换为词向量
        let x = self.embedding.forward(tokens);

        //将词向量编入transformer编码器
        let x = self.transformer.forward(TransformerEncoderInput::new(x));

        //平均池化
        let x = x.mean_dim(1);

        //输出，通过线性层
        self.output_layer.forward(x).squeeze(1)
    }
}

impl<B: AutodiffBackend> TrainStep<ClassificationBatch<B>,RegressionOutput<B>> for TextRegressionModel<B>{
    fn step(&self, batch: ClassificationBatch<B>) -> TrainOutput<RegressionOutput<B>> {    
        
        let tokens = batch.tokens;
        let targets = batch.targets;

        //进行前向传播
        let output = self.forward(tokens);

        //计算损失
        //使用均方误差
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Mean);

        let grads = loss.backward();

        TrainOutput::new(self, grads, RegressionOutput::new(loss, output, targets))
    }
}

impl<B:Backend> ValidStep<ClassificationBatch<B>,RegressionOutput<B>> for TextRegressionModel<B>{
    fn step(&self, batch: ClassificationBatch<B>) -> RegressionOutput<B> {
        let tokens = batch.tokens;
        let targets = batch.targets;

        let output = self.forward(tokens);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Mean);

        RegressionOutput{
            loss,output,targets
        }
    }
}