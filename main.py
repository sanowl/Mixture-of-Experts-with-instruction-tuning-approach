import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
from transformers import PreTrainedTokenizer
import json

class InstructionTuningDataset(Dataset):
    """Dataset for instruction tuning with various tasks."""
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Format instruction and input
        instruction = item["instruction"]
        input_text = item.get("input", "")
        target = item["target"]
        
        # Combine instruction and input
        if input_text:
            full_prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
        else:
            full_prompt = f"Instruction: {instruction}\nOutput:"

#Tokziner input
        inputs  = self.tokenizer(
            full_prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
# Tokenize target
        targets = self.tokenizer(
            target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return{
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0),
            "task_id": item.get("task_id", 0)
        }
    
def prepare_instruction_data(
        task_files: List[str],
        tokenizer:PreTrainedTokenizer
) -> InstructionTuningDataset:
    """ Prepare instrucntion tunning data"""
    all_data = []

    for task_id, file_path in enumerate(task_files):
        with open(file_path, 'r') as f:
            task_data = json.load(f)
            
        # Add task_id to each example
        for item in task_data:
            item["task_id"] = task_id
            all