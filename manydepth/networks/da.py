import math
from .depth_anything_v2.dpt import DepthAnythingV2
import torch.nn as nn
import torch
from .depth_anything_v2.util.blocks import FeatureFusionBlock, _make_scratch
import copy
from .feature_fusion import MultiFrameFeatureFusion

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'in_channels': 384, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'in_channels': 768, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'in_channels': 1024, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'in_channels': 1536, 'out_channels': [1536, 1536, 1536, 1536]}
}


def get_da_encoder_decoder(encoder_name='vits', checkpoint=True):
    cfg = MODEL_CONFIGS[encoder_name]
    da_model = DepthAnythingV2(
        encoder=cfg['encoder'],
        features=cfg['features'],
        out_channels=cfg['out_channels']
    )
    if checkpoint:
        da_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder_name}.pth', map_location='cpu'))
    encoder = da_model.pretrained
    setattr(encoder, 'intermediate_layer_idx', da_model.intermediate_layer_idx[encoder_name])
    decoder = da_model.depth_head
    return copy.deepcopy(encoder), copy.deepcopy(decoder)


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)
    
class ManyDepthAnythingEncoder(nn.Module):
    def __init__(self, encoder_name='vits', checkpoint=True):
        super(ManyDepthAnythingEncoder, self).__init__()
        self.encoder = get_da_encoder_decoder(encoder_name, checkpoint)[0]
        
    def forward(self, image, lookup_frames, return_class_token=True):
        out_features = self.encoder.get_intermediate_layers(image, self.encoder.intermediate_layer_idx, return_class_token=return_class_token)
        b, n, c, h, w = lookup_frames.shape
        lookup_frames = lookup_frames.reshape((b*n, c, h, w))
        lookup_features = self.encoder.get_intermediate_layers(lookup_frames, self.encoder.intermediate_layer_idx, return_class_token=return_class_token)

        return out_features, lookup_features

class ManyDepthAnythingDecoder(nn.Module):
    def __init__(
        self, 
        in_channels=384, 
        features=64, 
        use_bn=False, 
        out_channels=[48, 96, 192, 384], 
        use_clstoken=False,
        matching_height=518//14,
        matching_width=518//14
    ):
        super(ManyDepthAnythingDecoder, self).__init__()
        self.num_ch_enc = in_channels
        self.matching_height = matching_height
        self.matching_width = matching_width
        self.out_channels = out_channels
        self.use_clstoken = use_clstoken
                
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        

        
        self.multi_frame_feature_fusion = nn.ModuleList([
            MultiFrameFeatureFusion(in_channels, self.matching_height, self.matching_width)
            for _ in range(1)
        ])
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
        )
        
    
    def _fuse_features_multi_frame(self, current_feats, lookup_feats, i):
        batch_size, channels, height, width = current_feats.shape
        
        # Reshape current_feats
        current_feats_flat = current_feats.view(batch_size, channels, -1)
        
        # Reshape lookup_feats
        lookup_feats_flat = lookup_feats.view(batch_size, -1, height * width)
        
        #concatenate features along channel dim
        fused_features = torch.cat((current_feats_flat, lookup_feats_flat), 1).permute(0,2,1)
        output = self.multi_frame_feature_fusion[i](fused_features).permute(0,2,1)
        output = output.view(batch_size, channels, height, width)

        return output

    def forward(self, out_features, lookup_features, patch_h, patch_w):
        out = []
        for i, (x, lookup_feature) in enumerate(zip(out_features, lookup_features)):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
                
                lookup_feature, cls_token = lookup_feature[0], lookup_feature[1]
                readout = cls_token.unsqueeze(1).expand_as(lookup_feature)
                lookup_feature = self.readout_projects[i](torch.cat((lookup_feature, readout), -1))
            else:
                x = x[0]
                lookup_feature = lookup_feature[0]
                
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            lookup_feature = lookup_feature.permute(0, 2, 1).reshape((lookup_feature.shape[0], lookup_feature.shape[-1], patch_h, patch_w))
            if i > -1:
                x = self._fuse_features_multi_frame(x, lookup_feature, 0)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out_ = nn.functional.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        depth = self.scratch.output_conv2(out_)
        # Normalize depth
        return depth, out_

