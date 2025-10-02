use anyhow::Result;
use jieba_rs::Jieba;
use serde::Deserialize;
use std::fs;

use crate::tensor::{Vocab, MAX_SEQ_LEN, PAD_ID};

use burn::data::dataset::{Dataset}; 

pub struct MyDataset {
    data: Vec<ProcessedItem>,
}

impl Dataset<ProcessedItem> for MyDataset {
    fn get(&self, index: usize) -> Option<ProcessedItem> {
        self.data.get(index).cloned() 
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

// 可选：一个从 Vec<ProcessedItem> 创建 MyDataset 的辅助函数
impl MyDataset {
    pub fn new(data: Vec<ProcessedItem>) -> Self {
        Self { data }
    }
}

#[derive(Debug, Deserialize)]
struct CsvType {
    id: u32,
    text: String,
    score: i32,
}

struct TextScoreItem {
    content: String,
    score: u32,
}

struct DatasetItem {
    id: u32,
    item: TextScoreItem,
}

pub struct TextClassificationBatcher {
    dataset: Vec<DatasetItem>,
}

pub struct TokenizedDataset {
    pub token_squences: Vec<Vec<String>>,
    pub socres: Vec<u32>,
}

#[derive(Clone,Debug)]
pub struct ProcessedItem{
    pub token_ids: Vec<u32>,
    pub score: u32,
}

impl TextClassificationBatcher {
    pub fn read_from_csv(path: String) -> Result<Self> {
        let content = fs::read_to_string(path).expect("Cannot Read From The File");
        let mut reader = csv::Reader::from_reader(content.as_bytes());

        let mut text_class_batcher = TextClassificationBatcher {
            dataset: Vec::new(),
        };
        for record in reader.deserialize() {
            let record: CsvType = record?;

            if record.score == -1 {
                continue;
            }

            let score_item = TextScoreItem {
                content: record.text,
                score: record.score as u32,
            };

            let data_item = DatasetItem {
                id: record.id,
                item: score_item,
            };
            text_class_batcher.dataset.push(data_item);
        }

        Ok(text_class_batcher)
    }

    pub fn tokenize(self) -> TokenizedDataset {
        let jieba = Jieba::new();

        let mut token_sequences = Vec::new();
        let mut scores = Vec::new();

        println!("Starting tokenization....");

        for data_item in self.dataset {
            let content = &data_item.item.content;

            let tokens = jieba
                .cut(content, false)
                .into_iter()
                .map(|s| s.to_owned())
                .collect();

            token_sequences.push(tokens);
            scores.push(data_item.item.score);
        }

        println!("Tokenization Completed.");

        TokenizedDataset {
            token_squences: token_sequences,
            socres: scores,
        }
    }
}

impl TokenizedDataset {
    //将token转换为ID序列，并进行填充和截断
    pub fn to_processed_sequences(&self, vocab: &Vocab) -> Vec<(Vec<u32>, u32)> {
        let mut processed_data = Vec::new();

        for (tokens, score) in self.token_squences.iter().zip(self.socres.iter()) {
            let mut id_sequence: Vec<u32> = tokens.iter().map(|t| vocab.get_id(t)).collect();

            //截断
            if id_sequence.len() > MAX_SEQ_LEN {
                id_sequence.truncate(MAX_SEQ_LEN);
            }

            //填充
            if id_sequence.len() < MAX_SEQ_LEN{
                //let padding_needed = MAX_SEQ_LEN - id_sequence.len();
                id_sequence.resize(MAX_SEQ_LEN, PAD_ID);
            }

            processed_data.push((id_sequence,*score));
        }

        processed_data
    }
}
