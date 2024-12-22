import unittest
import torch
from model.model import FLANMoEModel, MoELayer
from model.config import create_flan_moe_32b
from transformers import PreTrainedTokenizer
import tempfile
import os

class TestFLANMoE(unittest.TestCase):
    def setUp(self):
        self.config = create_flan_moe_32b()
        self.model = FLANMoEModel(self.config)
        self.batch_size = 4
        self.seq_length = 128
        
    def test_moe_layer(self):
        """Test MoE layer forward pass"""
        layer = MoELayer(self.config)
        hidden_states = torch.randn(
            self.batch_size, 
            self.seq_length, 
            self.config.hidden_size
        )
        
        output = layer(hidden_states)
        self.assertEqual(
            output.shape, 
            (self.batch_size, self.seq_length, self.config.hidden_size)
        )

    def test_forward_pass(self):
        """Test full model forward pass"""
        input_ids = torch.randint(
            0, self.config.vocab_size, 
            (self.batch_size, self.seq_length)
        )
        attention_mask = torch.ones(self.batch_size, self.seq_length)
        labels = torch.randint(
            0, self.config.vocab_size, 
            (self.batch_size, self.seq_length)
        )
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        self.assertIn("logits", outputs)
        self.assertIn("loss", outputs)
        self.assertEqual(
            outputs["logits"].shape,
            (self.batch_size, self.seq_length, self.config.vocab_size)
        )

    def test_model_save_load(self):
        """Test model saving and loading"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save model
            self.model.save_pretrained(tmp_dir)
            
            # Load model
            loaded_model = FLANMoEModel.from_pretrained(tmp_dir)
            
            # Compare outputs
            input_ids = torch.randint(
                0, self.config.vocab_size, 
                (self.batch_size, self.seq_length)
            )
            
            with torch.no_grad():
                original_output = self.model(input_ids)["logits"]
                loaded_output = loaded_model(input_ids)["logits"]
                
            torch.testing.assert_close(original_output, loaded_output)

    def test_expert_routing(self):
        """Test that experts are being routed correctly"""
        layer = MoELayer(self.config)
        hidden_states = torch.randn(
            self.batch_size, 
            self.seq_length, 
            self.config.hidden_size
        )
        
        # Force specific routing by manipulating router weights
        with torch.no_grad():
            layer.router.weight.zero_()
            layer.router.weight[0].fill_(1.0)  # Route everything to first expert
            
        output = layer(hidden_states)
        self.assertTrue(torch.all(output != 0))  # Ensure output is non-zero

    def test_input_validation(self):
        """Test input validation and error handling"""
        with self.assertRaises(ValueError):
            # Test with invalid input shape
            invalid_input = torch.randn(self.batch_size, self.config.hidden_size)
            self.model(input_ids=invalid_input)

if __name__ == '__main__':
    unittest.main() 