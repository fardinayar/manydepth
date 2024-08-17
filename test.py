from manydepth.networks.da import ManyDepthAnythingDecoder, get_da_encoder_decoder
import torch

def test_many_depth_anything_decoder():
    # Initialize the encoder and decoder
    decoder = ManyDepthAnythingDecoder()
    encoder, _ = get_da_encoder_decoder()

    # Create dummy data
    batch_size = 2
    channels = 3
    height = 518
    width = 518
    
    out_features = encoder.get_intermediate_layers(torch.rand(batch_size, channels, height, width), encoder.intermediate_layer_idx, return_class_token=True)
    lookup_features = encoder.get_intermediate_layers(torch.rand(batch_size, channels, height, width), encoder.intermediate_layer_idx, return_class_token=True)
    poses = torch.randn(batch_size, 2, 4, 4)
    K = torch.randn(batch_size, 4, 4)
    invK = torch.randn(batch_size, 4, 4)

    # Forward pass
    output = decoder(out_features, lookup_features, height//14, width//14, poses, K, invK)

    # Check the output shape
    assert output.shape == (batch_size, 1, height, width), f"Unexpected output shape: {output.shape}"

    print("Test passed!")

if __name__ == '__main__':
    test_many_depth_anything_decoder()