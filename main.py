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