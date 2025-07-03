import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiFrameFeatureFusion(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, matching_height, matching_width):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.matching_height = matching_height
        self.matching_width = matching_width

        # Shared input projection for both inputs
        self.input_proj = nn.Sequential(nn.Linear(input_dim, latent_dim))

        # Project latents to output dimension
        self.output_proj = nn.Sequential(nn.Linear(output_dim*2, output_dim),
                                         nn.ReLU(),
                                         nn.Linear(output_dim, output_dim))
        

        self.cost_fusion = nn.Sequential(nn.Linear(output_dim*2, output_dim),
                                         nn.ReLU(),
                                         nn.Linear(output_dim, output_dim))
        
        # Flow embedding for 2D flow vectors (dx, dy)
        # We'll use a 2D embedding that can handle flow vectors
        self.flow_embedding = nn.Linear(2, output_dim)  # Embed 2D flow vectors (dx, dy)
        
        self.layer_norm_1 = nn.LayerNorm(output_dim)
        self.layer_norm_2 = nn.LayerNorm(output_dim)

    def forward(self, input):
        """
        x1 and x2 are expected to be of shape (B, N, input_dim)
        where B is batch size and N is the number of tokens.
        N should be equal to matching_height * matching_width
        """
        x1, x2 = input[:, :, :self.input_dim], input[:, :, self.input_dim:]
        b, n, _ = x1.shape
        
        # Verify that N matches the expected spatial dimensions
        assert n == self.matching_height * self.matching_width, f"Expected N={self.matching_height * self.matching_width}, got N={n}"
        
        # 1. Project both inputs to the latent dimension
        x1_proj = self.input_proj(x1)  # shape: (B, N, latent_dim)
        x2_proj = self.input_proj(x2)  # shape: (B, N, latent_dim)

        # 2. Compute the implicit cost volume via differentiable similarity (dot-product)
        # The cost volume represents similarity between every token of x1 and x2.
        cost_volume = torch.matmul(x1_proj, x2_proj.transpose(1, 2))  # shape: (B, N, N)
        
        # Normalize similarities to get a soft matching distribution.
        cost_volume = F.softmax(cost_volume, dim=-1)

        # Compute 2D flow in spatial token space
        # For each token in x1, find the best match in x2 (index in flattened space)
        best_match_idx = cost_volume.argmax(dim=-1)  # (B, N)
        
        # Create 2D coordinate grids for original and matched positions
        # Original positions: (0,0), (0,1), ..., (H-1, W-1)
        orig_y, orig_x = torch.meshgrid(
            torch.arange(self.matching_height, device=best_match_idx.device),
            torch.arange(self.matching_width, device=best_match_idx.device),
            indexing='ij'
        )
        orig_positions = torch.stack([orig_y.flatten(), orig_x.flatten()], dim=-1)  # (N, 2)
        
        # Matched positions: convert flattened indices back to 2D coordinates
        matched_y = best_match_idx // self.matching_width  # (B, N)
        matched_x = best_match_idx % self.matching_width   # (B, N)
        matched_positions = torch.stack([matched_y, matched_x], dim=-1)  # (B, N, 2)
        
        # Compute 2D flow vectors (dy, dx)
        orig_positions_expanded = orig_positions.unsqueeze(0).expand(b, -1, -1)  # (B, N, 2)
        flow_2d = matched_positions - orig_positions_expanded  # (B, N, 2)
        
        # Convert flow_2d to float for the linear layer
        flow_2d = flow_2d.float()
        
        # Embed the 2D flow vectors
        flow_2d_emb = self.flow_embedding(flow_2d)  # (B, N, output_dim)

        # 3. Aggregate features from x2 based on the computed cost volume
        # For each token in x1, we obtain a weighted sum of features from x2.
        aggregated_features = self.output_proj(torch.cat([torch.matmul(cost_volume, x2), x1], dim=-1)) + x1
        out = self.layer_norm_1(aggregated_features)
        
        out = self.cost_fusion(torch.cat([out, flow_2d_emb], -1)) + out
        out = self.layer_norm_2(out)

        return out

