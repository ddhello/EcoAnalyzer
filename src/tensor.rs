use std::collections::HashMap;

use crate::dataset::TokenizedDataset;

const UNK_TOKEN: &str = "<unk>"; // Unknown Token
const PAD_TOKEN: &str = "<pad>"; // Padding Token

pub const PAD_ID: u32 = 0;
const UNK_ID: u32 = 1;

pub const MAX_SEQ_LEN: usize = 128; 

pub struct Vocab{
    token_to_id: HashMap<String,u32>,
    id_to_token: Vec<String>,
}

impl Vocab{
    fn new() -> Self{
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();

        //注册特殊token
        token_to_id.insert(PAD_TOKEN.to_string(), PAD_ID);
        id_to_token.push(PAD_TOKEN.to_string());

        token_to_id.insert(UNK_TOKEN.to_string(), UNK_ID);
        id_to_token.push(UNK_TOKEN.to_string());

        Self {
            token_to_id,
            id_to_token,
        }
    }

    pub fn build(dataset: &TokenizedDataset,min_freq: usize) ->Self{
        let mut vocab = Self::new();
        let mut token_counts:HashMap<String,usize> = HashMap::new();

        for tokens in &dataset.token_squences{
            for token in tokens{
                *token_counts.entry(token.clone()).or_insert(0) += 1;
            }
        }

        let mut next_id = vocab.id_to_token.len() as u32;

        for (token,count) in token_counts{
            if count >= min_freq{
                vocab.token_to_id.insert(token.clone(), next_id);
                vocab.id_to_token.push(token);
                next_id+=1;
            }
        }
        
        println!("Vacob Built. Total Size: {}",vocab.id_to_token.len());

        vocab
    }

    //获取token对应的id
    pub fn get_id(&self,token: &str) -> u32{
        *self.token_to_id.get(token).unwrap_or(&UNK_ID)
    }

    //获取词汇表大小
    pub fn len(&self) -> usize{
        self.id_to_token.len()
    }
}