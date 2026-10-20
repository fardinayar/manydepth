import torch
import torch.nn as nn
import loralib as lora

# Maps a user-facing encoder LoRA target name to the substrings used to match
# nn.Linear child module names inside a DINOv2 block.
#   "mlp"       -> Mlp.fc1 / Mlp.fc2
#   "attention" -> Attention.qkv / Attention.proj
ENCODER_LORA_TARGET_SUBSTRINGS = {
    "mlp": ("fc",),
    "attention": ("qkv", "proj"),
}


def encoder_lora_target_substrings(targets):
    """Resolve a list of target names (e.g. ["mlp", "attention"]) to the flat
    tuple of name substrings replace_linear_with_lora should match on."""
    unknown = sorted(set(targets) - ENCODER_LORA_TARGET_SUBSTRINGS.keys())
    if unknown:
        raise ValueError(
            "Unknown lora_encoder_targets entries: {}. Valid values: {}".format(
                unknown, sorted(ENCODER_LORA_TARGET_SUBSTRINGS)
            )
        )
    substrings = tuple(s for t in targets for s in ENCODER_LORA_TARGET_SUBSTRINGS[t])
    if not substrings:
        raise ValueError("lora_encoder_targets must not be empty when LoRA is enabled")
    return substrings


def replace_linear_with_lora(model, r=4, lora_alpha=4, lora_dropout=0.0, target_names=("fc",)):
    """
    Recursively replace nn.Linear layers whose child name contains any of
    target_names with LoRA linear layers.

    Args:
    - model: The PyTorch model to modify
    - r: LoRA rank
    - lora_alpha: LoRA alpha parameter
    - lora_dropout: Dropout probability for LoRA layers
    - target_names: substrings to match against child module names (case-insensitive)

    Returns:
    - The modified model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(t in name.lower() for t in target_names):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None

            # Create a new LoRA linear layer
            new_layer = lora.Linear(
                in_features=in_features,
                out_features=out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )

            # Copy the weights and bias from the original layer
            with torch.no_grad():
                new_layer.weight.copy_(module.weight)
                if bias:
                    new_layer.bias.copy_(module.bias)

            # Replace the old layer with the new one
            setattr(model, name, new_layer)
            print(f"Replaced {name} with LoRA layer")
        else:
            # Recursively apply to child modules
            replace_linear_with_lora(module, r, lora_alpha, lora_dropout, target_names)

    return model



def replace_conv_with_loraconv(model, r=4, lora_alpha=4, lora_dropout=0.0):
    """
    Recursively replace all convolutional layers in the model with LoRA-enhanced convolutional layers.

    Args:
    - model: The PyTorch model to modify.
    - r: LoRA rank.
    - lora_alpha: LoRA alpha parameter.
    - lora_dropout: Dropout probability for LoRA layers.

    Returns:
    - The modified model.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # Create a new LoRA Conv2d layer
            new_layer = lora.Conv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size[0],
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )

            # Copy the weights and bias from the original layer
            with torch.no_grad():
                new_layer.weight.copy_(module.weight)
                if module.bias is not None:
                    new_layer.bias.copy_(module.bias)

            # Replace the old layer with the new one
            setattr(model, name, new_layer)
        else:
            # Recursively apply to child modules
            replace_conv_with_loraconv(module, r, lora_alpha, lora_dropout)

    return model
