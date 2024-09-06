import torch
import torch.nn as nn



def affine_invariant_loss(predictions, ground_truths, mask, return_intermediate=False):
    """
    Computes the affine-invariant loss for batched input.
    
    Args:
        predictions (torch.Tensor): Predicted depth maps, shape (B, H, W).
        ground_truths (torch.Tensor): Ground truth depth maps, shape (B, H, W).
        mask (torch.Tensor): Mask to indicate valid pixels, shape (B, H, W).
    
    Returns:
        torch.Tensor: Affine-invariant loss.
    """
    B, _, H, W = predictions.shape
    predictions = predictions.view(B, -1)  # Flatten to (B, H*W)
    ground_truths = ground_truths.view(B, -1)  # Flatten to (B, H*W)
    mask = mask.view(B, -1)
    # Compute t(d) and s(d) for each batch
    t_d = torch.median(ground_truths, dim=1, keepdim=True).values
    s_d = torch.mean(torch.abs(ground_truths - t_d), dim=1, keepdim=True)
    
    # Align predictions and ground truths
    d_hat_gt = (ground_truths - t_d) / s_d
    
    t_p = torch.median(predictions, dim=1, keepdim=True).values
    s_p = torch.mean(torch.abs(predictions - t_p), dim=1, keepdim=True)
    
    d_hat_pred = (predictions - t_p) / s_p
    
    # Compute the affine-invariant mean absolute error
    loss = torch.mean(torch.abs(d_hat_pred - d_hat_gt)*mask)
    if return_intermediate:
        return loss, d_hat_pred.view(B, _, H, W), d_hat_gt.view(B, _, H, W)
    return loss