import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import math

@dataclass
class FLANMoEConfig:
    """Configuration following paper specs"""
    vocab_size: int = 32000
    hidden_dim: int = 2048  # Base size from paper
    num_layers: int = 32    # As used in ST32B
    num_attention_heads: int = 32
    num_experts: int = 64   # Paper uses 64 experts per layer
    expert_dim: int = 8192  # 4x hidden_dim as in paper
    max_sequence_length: int = 2048
    num_expert_slots: int = 2  # k=2 activated experts per token
    expert_capacity_factor: float = 1.25
    dropout: float = 0.1
    attention_dropout: float = 0.1
    expert_dropout: float = 0.2
    router_z_loss_coef: float = 0.001
    router_aux_loss_coef: float = 0.01

class ExpertGatingNetwork(nn.Module):
    """Expert router using Top-2 gating with load balancing"""
    def __init__(self, config: FLANMoEConfig):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        # Initialize router weights using paper's method
        torch.nn.init.zeros_(self.router.weight)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            - routing_weights: [batch, seq_len, num_selected_experts]
            - expert_indices: [batch, seq_len, num_selected_experts]
            - router_logits: Raw router outputs for loss calculation
        """
        router_logits = self.router(hidden_states)
        
        # Get top-k experts per token
        routing_weights, expert_indices = torch.topk(
            F.softmax(router_logits, dim=-1),
            k=self.config.num_expert_slots,
            dim=-1
        )
        
        # Normalize weights
        routing_weights = F.normalize(routing_weights, p=1, dim=-1)
        
        return routing_weights, expert_indices, router_logits

class ExpertFFN(nn.Module):
    """Single expert feed-forward network"""
    def __init__(self, config: FLANMoEConfig):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_dim, config.expert_dim)
        self.dense_4h_to_h = nn.Linear(config.expert_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.expert_dropout)
        self.act = nn.GELU()
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class MoELayer(nn.Module):
    """Mixture of Experts layer with load balancing and auxiliary losses"""
    def __init__(self, config: FLANMoEConfig):
        super().__init__()
        self.config = config
        self.gate = ExpertGatingNetwork(config)
        self.experts = nn.ModuleList([ExpertFFN(config) for _ in range(config.num_experts)])
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        routing_weights, expert_indices, router_logits = self.gate(hidden_states)
        
        # Initialize expert outputs
        final_hidden_states = torch.zeros_like(hidden_states)
        
        # Expert computation
        capacity = int(self.config.expert_capacity_factor * sequence_length)
        expert_mask = torch.zeros(
            (batch_size, sequence_length, self.config.num_experts),
            device=hidden_states.device
        )
        
        for expert_idx in range(self.config.num_experts):
            # Find tokens routed to this expert
            selected_expert = (expert_indices == expert_idx).any(dim=-1)
            if not selected_expert.any():
                continue
                
            # Get expert inputs
            expert_input = hidden_states[selected_expert]
            expert_weights = routing_weights[selected_expert][
                expert_indices[selected_expert] == expert_idx
            ]
            
            # Expert forward pass
            expert_output = self.experts[expert_idx](expert_input)
            expert_output = expert_output * expert_weights.unsqueeze(-1)
            
            # Accumulate output
            final_hidden_states[selected_expert] += expert_output
            
            # Track routing for load balancing
            expert_mask[:, :, expert_idx] = selected_expert.float()
            
        # Calculate auxiliary losses
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Z-loss from ST-MoE paper
        z_loss = torch.mean(torch.log(1 + torch.exp(router_logits).sum(dim=-1)) ** 2)
        
        # Load balancing loss
        expert_usage = expert_mask.sum(dim=(0, 1))
        target_usage = torch.ones_like(expert_usage) * expert_mask.sum() / self.config.num_experts
        balance_loss = torch.mean((expert_usage - target_usage) ** 2)
        
        aux_losses = {
            "router_z_loss": z_loss * self.config.router_z_loss_coef,
            "router_balance_loss": balance_loss * self.config.router_aux_loss_coef
        }
        
        return final_hidden_states, aux_losses

class FLANMoETransformerBlock(nn.Module):
    """Transformer block with MoE FFN on alternating layers"""
    def __init__(self, config: FLANMoEConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_moe = layer_idx % 2 == 1  # MoE on odd layers
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # FFN (MoE or dense)
        if self.use_moe:
            self.ffn = MoELayer(config)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_dim, config.expert_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.expert_dim, config.hidden_dim),
                nn.Dropout(config.dropout)
            )
            
        # Layer norms
        self.pre_attention_norm = nn.LayerNorm(config.hidden_dim)
        self.pre_ffn_norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        # Pre-norm architecture as used in paper
        normed_states = self.pre_attention_norm(hidden_states)
        attention_output, _ = self.attention(
            normed_states, normed_states, normed_states,
            key_padding_mask=attention_mask
        )
        hidden_states = hidden_states + attention_output
        
        normed_states = self.pre_ffn_norm(hidden_states)
        if self.use_moe:
            ffn_output, aux_losses = self.ffn(normed_states)
            hidden_states = hidden_states + ffn_output
            return hidden_states, aux_losses
        else:
            hidden_states = hidden_states + self.ffn(normed_states)
            return hidden_states, None

class FLANMoEModel(nn.Module):
    """Complete FLAN-MoE model with instruction tuning"""
    def __init__(self, config: FLANMoEConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Position embeddings
        self.register_buffer(
            "position_embeddings",
            self._create_position_embeddings(config.max_sequence_length, config.hidden_dim)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            FLANMoETransformerBlock(config, i)
            for i in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.vocab_size)
        
        self._init_weights()
        
    def _create_position_embeddings(self, max_length: int, hidden_dim: int) -> torch.Tensor:
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim)
        )
        pos_emb = torch.zeros(1, max_length, hidden_dim)
        pos_emb[0, :, 0::2] = torch.sin(position * div_term)
        pos_emb[0, :, 1::2] = torch.cos(position * div_term)
        return pos_emb
        
    def _init_weights(self):
        """Initialize weights following paper specifications"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Get embeddings
        hidden_states = self.embeddings(input_ids)
        hidden_states = hidden_states + self.position_embeddings[:, :hidden_states.size(1)]
        
        # Track auxiliary losses
        all_aux_losses = []
        
        # Forward through layers
        for layer in self.layers:
            hidden_states, aux_losses = layer(hidden_states, attention_mask)
            if aux_losses is not None:
                all_aux_losses.append(aux_losses)
        
        # Final layer norm and output projection
        hidden_states = self.final_norm(hidden_states)
        logits = self.output(hidden_states)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            # Main language modeling loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            outputs["task_loss"] = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            # Combine auxiliary losses
            if all_aux_losses:
                aux_loss = sum(
                    sum(losses.values()) 
                    for losses in all_aux_losses
                )
                outputs["aux_loss"] = aux_loss
                outputs["total_loss"] = outputs["task_loss"] + aux_loss
            else:
                outputs["total_loss"] = outputs["task_loss"]
        
        return outputs

def create_flan_moe_32b():
    """Creates FLAN-MoE-32B configuration from paper"""
    return FLANMoEConfig(
        hidden_dim=4096,
        num_layers=32,
        num_attention_heads=64,
        num_experts=64,
        expert_dim=16384
    )

def create_flan_moe_base():
    """Creates FLAN-MoE-Base configuration from paper"""
    return FLANMoEConfig(
        hidden_dim=1024,
        num_layers=12,
        num_attention_heads=16,
        num_experts=64,
        expert_dim=4096
    )