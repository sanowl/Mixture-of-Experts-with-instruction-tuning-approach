import json
import os
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
import hashlib
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process_raw_data(self, task_name: str, raw_data: List[Dict]) -> List[Dict]:
        """Process raw data into the format needed for training"""
        processed_data = []
        
        for item in raw_data:
            # Ensure required fields exist
            if not all(k in item for k in ["instruction", "response"]):
                continue
                
            # Clean and validate text
            instruction = self._clean_text(item["instruction"])
            response = self._clean_text(item["response"])
            
            if not self._validate_example(instruction, response):
                continue
            
            # Create processed example
            processed_example = {
                "instruction": instruction,
                "response": response,
                "task_name": task_name,
                "id": self._create_example_id(instruction, response)
            }
            
            # Add optional fields if they exist
            for field in ["metadata", "category", "difficulty"]:
                if field in item:
                    processed_example[field] = item[field]
            
            processed_data.append(processed_example)
        
        return processed_data
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
            
        # Basic cleaning
        text = text.strip()
        text = " ".join(text.split())  # Normalize whitespace
        text = text.replace("\t", " ")
        
        return text
    
    def _validate_example(self, instruction: str, response: str) -> bool:
        """Validate a single example"""
        # Check for empty strings
        if not instruction or not response:
            return False
            
        # Check lengths
        if len(instruction.split()) > 512 or len(response.split()) > 512:
            return False
            
        # Check for low-quality examples
        if len(instruction.split()) < 3 or len(response.split()) < 3:
            return False
            
        return True
    
    def _create_example_id(self, instruction: str, response: str) -> str:
        """Create a unique ID for an example"""
        content = f"{instruction}{response}".encode('utf-8')
        return hashlib.md5(content).hexdigest()
    
    def process_file(self, input_file: Path) -> None:
        """Process a single input file"""
        task_name = input_file.stem
        output_file = self.output_dir / f"{task_name}.json"
        
        try:
            # Load raw data
            with open(input_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Process data
            processed_data = self.process_raw_data(task_name, raw_data)
            
            # Save processed data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processed {len(processed_data)} examples for {task_name}")
            
        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}")
    
    def process_all_files(self, num_workers: int = 4):
        """Process all files in input directory"""
        input_files = list(self.input_dir.glob("*.json"))
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(
                executor.map(self.process_file, input_files),
                total=len(input_files),
                desc="Processing files"
            ))
    
    def validate_dataset(self, data_path: str) -> bool:
        """Validate that dataset follows required format"""
        try:
            data_file = Path(data_path)
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.error("Data must be a list of examples")
                return False
            
            # Validate each example
            for idx, example in enumerate(data):
                if not isinstance(example, dict):
                    logger.error(f"Example {idx} must be a dictionary")
                    return False
                
                # Check required fields
                required_fields = ["instruction", "response", "task_name", "id"]
                for field in required_fields:
                    if field not in example:
                        logger.error(f"Example {idx} missing required field: {field}")
                        return False
                
                # Validate field types
                if not all(isinstance(example[f], str) for f in required_fields):
                    logger.error(f"Example {idx} has invalid field types")
                    return False
            
            logger.info(f"Dataset {data_path} is valid with {len(data)} examples")
            return True
            
        except Exception as e:
            logger.error(f"Error validating dataset: {e}")
            return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--validate", type=str, help="Path to validate a processed dataset")
    args = parser.parse_args()
    
    if args.validate:
        processor = DataProcessor("", "")
        processor.validate_dataset(args.validate)
    else:
        processor = DataProcessor(args.input_dir, args.output_dir)
        processor.process_all_files(args.num_workers)

if __name__ == "__main__":
    main()