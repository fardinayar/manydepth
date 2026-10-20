import os
from .depth_anything_v2.dpt import DepthAnythingV2
import torch.nn as nn
import torch
import torch.nn.functional as F
from .depth_anything_v2.util.blocks import FeatureFusionBlock, _make_scratch
import copy

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'in_channels': 384, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'in_channels': 768, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'in_channels': 1024, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'in_channels': 1536, 'out_channels': [1536, 1536, 1536, 1536]}
}


def get_da_encoder_decoder(encoder_name='vits', checkpoint=True, checkpoint_dir='checkpoints'):
    cfg = MODEL_CONFIGS[encoder_name]
    da_model = DepthAnythingV2(
        encoder=cfg['encoder'],
        features=cfg['features'],
        out_channels=cfg['out_channels']
    )
    if checkpoint:
        checkpoint_path = os.path.join(checkpoint_dir, f'depth_anything_v2_{encoder_name}.pth')
        da_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
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


class ClsTokenScaleShiftCorrector(nn.Module):
    """Predict disparity affine scale*disp+shift from the concatenated per-layer CLS tokens.

    All num_layers CLS tokens are concatenated and squeezed through a low-dim bottleneck
    before the scale/shift heads. The bottleneck regularizes the head (which regresses only
    2 global scalars per image) and keeps it smaller than the previous per-layer-weighted
    MLP, without needing an extra regularization loss term.
    """

    def __init__(self, embed_dim, num_layers=4, bottleneck_dim=128, dropout=0.1):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_layers = num_layers
        in_dim = self.embed_dim * self.num_layers

        h = bottleneck_dim

        self.shared_mlp = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.GELU(),
            
        )

        self.scale_head = nn.Linear(h, 1)
        self.shift_head = nn.Linear(h, 1)

        with torch.no_grad():
            self.scale_head.bias[0] = torch.log(
                torch.exp(torch.tensor(1.0)) - 1.0
            ).item()

    def forward(self, cls_tokens):
        # cls_tokens: [B, num_layers, embed_dim] -> concat to [B, num_layers * embed_dim]
        token = cls_tokens.reshape(cls_tokens.shape[0], -1)

        shared = self.shared_mlp(token)
        raw_scale = self.scale_head(shared)
        raw_shift = self.shift_head(shared)

        scale = F.softplus(raw_scale).unsqueeze(-1).unsqueeze(-1) + 1e-6
        shift = raw_shift.unsqueeze(-1).unsqueeze(-1)
        return scale, shift


class ManyDepthAnythingEncoder(nn.Module):
    def __init__(self, encoder_name='vits', checkpoint=True, checkpoint_dir='checkpoints'):
        super(ManyDepthAnythingEncoder, self).__init__()
        self.encoder = get_da_encoder_decoder(encoder_name, checkpoint, checkpoint_dir)[0]
        
    def forward(self, image, return_class_token=True):
        return self.encoder.get_intermediate_layers(
            image, self.encoder.intermediate_layer_idx, return_class_token=return_class_token
        )

class ManyDepthAnythingDecoder(nn.Module):
    def __init__(
        self,
        in_channels=384,
        features=64,
        use_bn=False,
        out_channels=[48, 96, 192, 384],
        use_clstoken=False,
        patch_h=518//14,
        patch_w=518//14,
        use_cls_scale_shift=False,
    ):
        super(ManyDepthAnythingDecoder, self).__init__()
        self.num_ch_enc = in_channels
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.out_channels = out_channels
        self.use_clstoken = use_clstoken
        self.use_cls_scale_shift = use_cls_scale_shift
        self.cls_scale_shift = (
            ClsTokenScaleShiftCorrector(in_channels) if use_cls_scale_shift else None
        )

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
    
    def forward(self, out_features):
        out = []
        cls_token_list = []
        for i, x in enumerate(out_features):
            if (
                self.cls_scale_shift is not None
                and isinstance(x, (tuple, list))
                and len(x) == 2
            ):
                cls_token_list.append(x[1])
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], self.patch_h, self.patch_w))
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
        out_ = nn.functional.interpolate(out, (int(self.patch_h * 14), int(self.patch_w * 14)), mode="bilinear", align_corners=True)
        depth = self.scratch.output_conv2(out_)
        if self.cls_scale_shift is not None and cls_token_list:
            # Per-image mean/std with detached stats: CLS affine on standardized maps (~O(1)) so
            # metric / reprojection loss gradients do not disproportionately drive scale vs shape.
            eps = 1e-6
            mu = depth.mean(dim=(2, 3), keepdim=True)
            std = depth.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(eps)
            dn = (depth - mu) / std
            stacked = torch.stack(cls_token_list, dim=1).to(dtype=depth.dtype)  # [B, num_layers, embed_dim]
            scale, shift = self.cls_scale_shift(stacked)
            depth = depth * scale + shift
        return depth, out_
