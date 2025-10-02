use crate::{ trian::run_train_setup};
use anyhow::{Ok, Result};

mod dataset;
mod tensor;
mod batcher;
mod trian;

fn main() -> Result<()>{
    run_train_setup()?;

    Ok(())
}
