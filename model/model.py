import torch
import torch.nn as nn
from transformers import T5PreTrainedModel
from typing import Optional, Dict, List
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Router
        self.router = nn.Linear(self.hidden_size, self.num_experts)
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.intermediate_size),
                nn.GELU(),
                nn.Linear(self.intermediate_size, self.hidden_size)
            ) for _ in range(self.num_experts)
        ])
        
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Router logistics
        router_logits = self.router(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, k=2, dim=-1
        )
        
        # Normalize weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        expert_outputs = torch.zeros_like(hidden_states)
        for i, expert in enumerate(self.experts):
            expert_mask = top_k_indices == i
            if expert_mask.any():
                expert_inputs = hidden_states[expert_mask]
                expert_outputs[expert_mask] = expert(expert_inputs)
        
        # Combine expert outputs
        final_output = torch.zeros_like(hidden_states)
        for i in range(2):
            expert_idx = top_k_indices[..., i]
            weight = top_k_weights[..., i, None]
            final_output += weight * expert_outputs
            
        return final_output

class MoEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            MoELayer(config) for _ in range(config.num_hidden_layers)
        ])
        
    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class MoEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            MoELayer(config) for _ in range(config.num_hidden_layers)
        ])
        
    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class FLANMoEModel(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Core transformer layers
        self.encoder = MoEEncoder(config)
        self.decoder = MoEDecoder(config)
        
        # Output layer
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.init_weights()
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Encode
        encoder_outputs = self.encoder(input_ids)
        
        # Decode
        decoder_outputs = self.decoder(encoder_outputs)
        
        # Language modeling head
        lm_logits = self.lm_head(decoder_outputs)
        
        outputs = {"logits": lm_logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
            outputs["loss"] = loss
            
        return outputs