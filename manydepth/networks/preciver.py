import torch
import torch.nn as nn
import torch.nn.functional as F

class PreceiverIO(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, num_latents, num_output_queries, num_heads, mlp_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_latents = num_latents
        self.num_output_queries = num_output_queries

        # Shared input projection for both inputs
        self.input_proj = nn.Linear(input_dim, latent_dim)

        # Learnable latent array (for further decoding)
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))

        # Output queries for decoder (if needed)
        self.output_queries = nn.Parameter(torch.randn(1, num_output_queries, output_dim))

        # Transformer decoder layer to refine aggregated features from cost volume
        self.decoder_layer = nn.TransformerDecoderLayer(output_dim, num_heads, mlp_dim, batch_first=True)

        # Project latents to output dimension
        self.output_proj = nn.Linear(latent_dim, output_dim)

        self.cost_fusion = nn.Sequential(nn.Linear(output_dim*2, output_dim),
                                         nn.ReLU(),
                                         nn.Linear(output_dim, output_dim))
        

    def forward(self, input):
        """
        x1 and x2 are expected to be of shape (B, N, input_dim)
        where B is batch size and N is the number of tokens.
        """
        x1, x2 = input[:, :, :self.input_dim], input[:, :, self.input_dim:]
        b, n, _ = x1.shape
        # 1. Project both inputs to the latent dimension
        x1_proj = self.input_proj(x1)  # shape: (B, N, latent_dim)
        x2_proj = self.input_proj(x2)  # shape: (B, N, latent_dim)

        # 2. Compute the implicit cost volume via differentiable similarity (dot-product)
        # The cost volume represents similarity between every token of x1 and x2.
        cost_volume = torch.matmul(x1_proj, x2_proj.transpose(1, 2))  # shape: (B, N, N)
        
        # Normalize similarities to get a soft matching distribution.
        cost_volume = F.softmax(cost_volume, dim=-1)


        cost_summary = cost_volume.mean(dim=-1, keepdim=True)  # shape: (B, N, 1)
        cost_summary = cost_summary.repeat(1, 1, self.output_dim)  # shape: (B, N, output_dim)


        # 3. Aggregate features from x2 based on the computed cost volume
        # For each token in x1, we obtain a weighted sum of features from x2.
        aggregated_features = torch.matmul(cost_volume, x2_proj)  # shape: (B, N, latent_dim)
        
        # Project aggregated features to output dimension before decoding.
        memory = self.output_proj(aggregated_features)  # shape: (B, N, output_dim)

        # Repeat output queries for each batch.
        output_queries = self.output_queries.repeat(b, 1, 1)  # shape: (B, num_output_queries, output_dim)

        # 5. Refine output queries using transformer decoder, where the memory is the aggregated feature context.
        out = self.decoder_layer(x1, memory)  # shape: (B, num_output_queries, output_dim)

        out = self.cost_fusion(torch.cat([out, cost_summary], -1)) + x1
        # Optionally add a residual connection with a slice of one of the inputs if applicable:
        # For instance, if output_dim <= input_dim, you might do:
        
        return out

