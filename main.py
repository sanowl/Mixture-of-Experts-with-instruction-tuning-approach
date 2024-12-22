import argparse
import logging
import os
from pathlib import Path
import torch
import wandb
from train import FLANTrainer, InstructionTuningDataset
from model.config import create_flan_moe_32b, FLANMoEModel
from transformers import T5Tokenizer
from data_processing import DataProcessor
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_local_data(data_path: str):
    """Load and validate local training data"""
    processor = DataProcessor("", "")
    data_files = Path(data_path).glob("*.json")
    all_data = []
    
    for file in data_files:
        if processor.validate_dataset(str(file)):
            with open(file, 'r') as f:
                data = json.load(f)
                all_data.extend(data)
                logger.info(f"Loaded {len(data)} examples from {file.name}")
    
    return all_data

def setup_training_environment(args):
    """Setup training environment and devices"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} GPU(s)")
        if device_count > 1:
            logger.info("Using DataParallel")
    else:
        logger.warning("No GPU found, using CPU")
    
    if args.use_wandb:
        wandb.init(
            project="flan-moe",
            name="flan-moe-local",
            config=vars(args)
        )

def main():
    parser = argparse.ArgumentParser(description="Train FLAN-MoE model locally")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--local_data_path", type=str, default="data/")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="models/")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()
    
    # Setup environment
    setup_training_environment(args)
    
    # Initialize model and tokenizer
    logger.info("Initializing model and tokenizer...")
    config = create_flan_moe_32b()
    model = FLANMoEModel(config)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Load and prepare data
    logger.info(f"Loading data from {args.local_data_path}")
    train_data = load_local_data(args.local_data_path)
    train_dataset = InstructionTuningDataset(train_data, tokenizer)
    
    # Initialize trainer
    trainer = FLANTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        config={
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
            "warmup_steps": args.warmup_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
    )
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        # Save final model
        output_path = Path(args.output_dir) / "final_model"
        trainer.save_model(str(output_path))
        logger.info(f"Model saved to {output_path}")

if __name__ == "__main__":
    main()