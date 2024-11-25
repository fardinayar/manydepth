import torch
import torch.nn as nn
import loralib as lora

def replace_qkv_with_mergedlinear(model, r=1, lora_alpha=32, lora_dropout=0.1):
    """
    Recursively replace all qkv linear layers in the model with MergedLinear from loralib.
    
    Args:
    - model: The PyTorch model to modify
    - r: LoRA rank
    - lora_alpha: LoRA alpha parameter
    - lora_dropout: Dropout probability for LoRA layers
    
    Returns:
    - The modified model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and module.out_features == module.in_features * 3 and 'qkv' in name.lower():
            # This is likely a qkv linear layer
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            
            # Create a new MergedLinear layer
            new_layer = lora.MergedLinear(
                in_features,
                out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                enable_lora=[True, True, True],  # Enable LoRA for q and v
                bias=bias,
                merge_weights=True,
            )
            
            # Copy the weights and bias from the original layer
            with torch.no_grad():
                new_layer.weight.copy_(module.weight)
                if bias:
                    new_layer.bias.copy_(module.bias)
            
            # Replace the old layer with the new one
            setattr(model, name, new_layer)
            print(f"Replaced layer: {name}")
        else:
            # Recursively apply to child modules
            replace_qkv_with_mergedlinear(module, r, lora_alpha, lora_dropout)
    
    return model