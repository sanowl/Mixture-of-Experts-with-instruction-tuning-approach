def create_flan_moe_32b():
    """Creates config for 32B FLAN-MoE model"""
    return {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_experts": 8,
        "expert_capacity": 32,
        "vocab_size": 32128,  # T5 vocab size
        "intermediate_size": 11008,
    } 