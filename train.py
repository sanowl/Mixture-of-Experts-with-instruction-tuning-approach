# training.py
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, get_scheduler
from typing import List, Dict, Any, Optional
import json
import logging
from tqdm import tqdm
import numpy as np
import wandb

class InstructionTuningDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: PreTrainedTokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item["instruction"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            item["response"],
            padding="max_length", 
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": labels.input_ids.squeeze()
        }

class FLANTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Dict[str, Any] = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        
        self.config = {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "warmup_steps": 10000,
            "max_steps": 200000,
            "gradient_accumulation_steps": 1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "logging_steps": 100,
            "eval_steps": 5000,
            "save_steps": 10000,
            **(config or {})
        }
        
        self._setup_training()

    def _setup_training(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_params = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_params,
            lr=self.config["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.config["warmup_steps"],
            num_training_steps=self.config["max_steps"]
        )
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        if self.eval_dataset:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=4
            )

    def train(self):
        self.model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        global_step = 0
        total_loss = 0
        best_eval_loss = float('inf')
        
        progress_bar = tqdm(total=self.config["max_steps"], desc="Training")
        train_iterator = iter(self.train_dataloader)
        
        while global_step < self.config["max_steps"]:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)
            
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs["loss"]
            
            if self.config["gradient_accumulation_steps"] > 1:
                loss = loss / self.config["gradient_accumulation_steps"]
            
            loss.backward()
            
            if (global_step + 1) % self.config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config["max_grad_norm"]
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            global_step += 1
            progress_bar.update(1)
            
            if global_step % self.config["logging_steps"] == 0:
                avg_loss = total_loss / self.config["logging_steps"]
                wandb.log({
                    "train/loss": avg_loss,
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/step": global_step
                })
                total_loss = 0
            
            if self.eval_dataset and global_step % self.config["eval_steps"] == 0:
                eval_loss = self.evaluate()
                wandb.log({
                    "eval/loss": eval_loss,
                    "eval/step": global_step
                })
                
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    self.save_model("best_model")
            
            if global_step % self.config["save_steps"] == 0:
                self.save_model(f"checkpoint-{global_step}")
        
        progress_bar.close()
        return global_step

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        eval_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs["loss"].item()
                eval_steps += 1
        
        return total_loss / eval_steps

    def save_model(self, output_dir: str):
        self.model.save_pretrained(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)


