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
        
        self.layer_norm_1 = nn.LayerNorm(output_dim)

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

        # 3. Aggregate features from x2 based on the computed cost volume
        # For each token in x1, we obtain a weighted sum of features from x2.
        aggregated_features = self.output_proj(torch.cat([torch.matmul(cost_volume, x2), x1], dim=-1)) + x1
        out = self.layer_norm_1(aggregated_features)

        return out

