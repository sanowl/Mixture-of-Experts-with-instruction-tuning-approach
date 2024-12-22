import torch
from train import FLANTrainer, InstructionTuningDataset
from model.config import create_flan_moe_32b, FLANMoEModel
from transformers import T5Tokenizer
import wandb
import os
import json
import argparse
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_from_s3(data_path: str) -> List[Dict]:
    """Load training data from S3 or local path"""
    all_data = []
    data_files = [
        "muffin.json",
        "t0_sf.json",
        "niv2.json",
        "cot.json"
    ]
    
    for filename in data_files:
        file_path = os.path.join(data_path, filename)
        try:
            if file_path.startswith('s3://'):
                import boto3
                s3 = boto3.client('s3')
                bucket = file_path.split('/')[2]
                key = '/'.join(file_path.split('/')[3:])
                response = s3.get_object(Bucket=bucket, Key=key)
                data = json.loads(response['Body'].read())
            else:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            
            # Add task name to each example
            task_name = os.path.splitext(filename)[0]
            for item in data:
                item["task_name"] = task_name
            all_data.extend(data)
            
            logger.info(f"Loaded {len(data)} examples from {filename}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
    
    return all_data

def setup_aws_training(
    batch_size: int = 32,
    data_path: str = "data/",
    output_path: str = "models/",
    use_wandb: bool = True
):
    """Setup and start training on AWS"""
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(
            project="flan-moe",
            name="flan-moe-32b-aws",
            config={
                "batch_size": batch_size,
                "instance_type": os.environ.get("AWS_INSTANCE_TYPE", "unknown")
            }
        )
    
    # Setup multi-GPU if available
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and config
    logger.info("Initializing model...")
    config = create_flan_moe_32b()
    model = FLANMoEModel(config)
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    # Load datasets
    logger.info(f"Loading data from {data_path}...")
    train_data = load_data_from_s3(data_path)
    train_dataset = InstructionTuningDataset(train_data, tokenizer)
    
    # Initialize trainer with AWS-specific config
    trainer = FLANTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        config={
            "batch_size": batch_size,
            "gradient_accumulation_steps": 4,  # Adjust based on GPU memory
            "save_steps": 1000,  # Save more frequently on AWS
            "logging_steps": 50
        }
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_output_path = os.path.join(output_path, "final_model")
    trainer.save_model(final_output_path)
    logger.info(f"Training complete. Model saved to {final_output_path}")
    
    # Sync to S3 if needed
    if output_path.startswith('s3://'):
        logger.info(f"Uploading model to {output_path}...")
        os.system(f"aws s3 sync {final_output_path} {output_path}/final_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--output_path", type=str, default="models/")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    
    setup_aws_training(
        batch_size=args.batch_size,
        data_path=args.data_path,
        output_path=args.output_path,
        use_wandb=not args.no_wandb
    ) 