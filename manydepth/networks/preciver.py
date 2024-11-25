import torch
import torch.nn as nn

class PreceiverIO(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, num_latents, num_output_queries, num_heads, num_layers, mlp_dim):
        super().__init__()
        self.lamda = nn.Parameter(torch.tensor(0.0)) 
        latent_dim = int(latent_dim)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_latents = num_latents
        self.num_output_queries = num_output_queries

        # Input projection
        self.input_proj = nn.Linear(input_dim, latent_dim)
        
        # Latent array
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))

        # Output queries
        self.output_queries = nn.Parameter(torch.randn(1, num_output_queries, output_dim))

        # Encoder (input to latents)
        self.encoder_cross_attn = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(latent_dim, num_heads, mlp_dim, batch_first=True)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(latent_dim, output_dim)
        # Decoder (latents to output)
        self.decoder_cross_attn = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
        self.decoder_layer = nn.TransformerDecoderLayer(output_dim, num_heads, mlp_dim, batch_first=True)
        

    def forward(self, x):
        b, n, _ = x.shape

        # Project input to latent dimension
        x_proj = self.input_proj(x)

        # Encode
        latents = self.latents.repeat(b, 1, 1)
        latents, _ = self.encoder_cross_attn(latents, x_proj, x_proj)
        for layer in self.encoder_layers:
            latents = layer(latents)

        # Decode
        latents = self.output_proj(latents)  # Project latents to output dimension

        output_queries = self.output_queries.repeat(b, 1, 1)
        out, _ = self.decoder_cross_attn(output_queries, latents, latents)
        out = self.decoder_layer(out, latents)

        
        lamda = torch.sigmoid(self.lamda)  # Sigmoid activation for lambda
        return (out * lamda + (1-lamda) * x[:, :, :self.output_dim])

'''# Example usage
input_dim = 256
latent_dim = 64
output_dim = 128
num_latents = 32
num_output_queries = 50
num_heads = 8
num_layers = 6
mlp_dim = 256

model = PreceiverIO(input_dim, latent_dim, output_dim, num_latents, num_output_queries, num_heads, num_layers, mlp_dim)
x = torch.randn(1, 100, input_dim)  # Batch size 1, 100 tokens, input dimension 256
output = model(x)
print(output.shape)  # Should be torch.Size([1, 50, 128])'''