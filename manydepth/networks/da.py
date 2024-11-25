import math
from .depth_anything_v2.dpt import DepthAnythingV2
import torch.nn as nn
from functools import partial
import torch
from .resnet_encoder import ResnetEncoderMatching
from layers import BackprojectDepth, Project3D
from .depth_anything_v2.util.blocks import FeatureFusionBlock, _make_scratch
import copy
from .preciver import PreceiverIO

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=384*2, dropout=0.9):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu
        self.pe = PositionalEncoding(d_model)
        self.lam = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, input_tgt, memory):
        tgt = self.pe(input_tgt)
        memory = self.pe(memory)
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(tgt2, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt*F.sigmoid(self.lam) + input_tgt*(1 - F.sigmoid(self.lam))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1369):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
class DepthScaler(nn.Module):    
    def __init__(self, in_channels=384, num_heads=8, dropout_rate=0.4, layern=True):
        super().__init__()
        
        self.query_scale_shift = nn.Parameter(torch.randn(1, 1, in_channels))
        self.attn_shift = nn.ModuleList(
            [nn.MultiheadAttention(in_channels, num_heads=num_heads, dropout=dropout_rate, batch_first=True) for _ in range(3)]
        )
        self.lin_shift = nn.ModuleList(
            [nn.Linear(in_channels, in_channels) for _ in range(3)]
        )
        
        self.layern = nn.ModuleList(
            [nn.Identity(in_channels) for _ in range(3)]   
        )
        self.layern_shift = nn.LayerNorm(in_channels)
        self.layern_scale = nn.LayerNorm(in_channels)

        self.scale_proj = nn.Linear(in_channels, 1)
        self.shift_proj = nn.Linear(in_channels, 1)
        
        self.pos_encoder = PositionalEncoding(in_channels)

    
    def forward(self, features, depth=None):
        # Combine features
        attn_output = self.query_scale_shift.expand(features[0][0].size(0), -1, -1)  # [1, B, in_channels]


        for i in range(3):
            x = features[i-3][0]
            x = self.pos_encoder(x)

            attn_output, _ = self.attn_shift[0](attn_output, x, x) 
            attn_output = self.lin_shift[0](attn_output+ attn_output) + attn_output
            # Project to get final scale and shift
        scale = self.scale_proj(self.layern_scale(attn_output)).unsqueeze(-1)
        shift = self.shift_proj(self.layern_shift(attn_output)).unsqueeze(-1)
        
        return torch.abs(scale), shift

def get_da_encoder_decoder(encoder_name='vits', checkpoint=True):
    da_model = DepthAnythingV2(**MODEL_CONFIGS[encoder_name])
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
        with torch.no_grad():
            lookup_features = self.encoder.get_intermediate_layers(lookup_frames, self.encoder.intermediate_layer_idx, return_class_token=return_class_token)

        return out_features, lookup_features

class ManyDepthAnythingDecoder(ResnetEncoderMatching):
    def __init__(
        self, 
        in_channels=384, 
        features=64, 
        use_bn=False, 
        out_channels=[48, 96, 192, 384], 
        use_clstoken=False,
        min_depth_bin=0.1,
        max_depth_bin=20.0,
        num_depth_bins=96,
        adaptive_bins=False,
        depth_binning='linear',
        matching_height=518//14,
        matching_width=518//14
    ):
        super(ResnetEncoderMatching, self).__init__()
        ### Add mathing to DPT
        self.adaptive_bins = adaptive_bins
        self.depth_binning = depth_binning
        self.set_missing_to_max = True
        self.num_depth_bins = num_depth_bins
        self.is_cuda = False
        self.warp_depths = None
        self.depth_bins = None
        self.num_ch_enc = in_channels
        self.matching_height = matching_height
        self.matching_width = matching_width
        self.out_channels = out_channels
        self._init_feature_matching_moudules(min_depth_bin, max_depth_bin)

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
        '''self.transformer_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=in_channels, nhead=8)
            for _ in range(1)  # You can adjust the number of layers
        ])'''
        

        
        self.preciver = PreceiverIO(in_channels*2, in_channels/4, in_channels, 96, self.matching_height*self.matching_width, 4, 4, 96*2)
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
        
    def _init_feature_matching_moudules(self, min_depth_bin, max_depth_bin):
        self.backprojector = BackprojectDepth(batch_size=self.num_depth_bins,
                                              height=self.matching_height,
                                              width=self.matching_width)
        self.projector = Project3D(batch_size=self.num_depth_bins,
                                   height=self.matching_height,
                                   width=self.matching_width)
        self.compute_depth_bins(min_depth_bin, max_depth_bin)

        self.reduce_convs = nn.ModuleList()
        for i in range(len(self.out_channels)):
            reduce_conv = nn.Sequential(nn.Conv2d(self.num_ch_enc + self.num_depth_bins,
                                                    out_channels=self.num_ch_enc,
                                                    kernel_size=3, stride=1, padding=1),
                                            nn.GELU(),
                                        nn.Conv2d(self.num_ch_enc,
                                                    out_channels=self.num_ch_enc,
                                                    kernel_size=3, stride=1, padding=1),
                                        )
            self.reduce_convs.append(reduce_conv)

    
    def _fuse_matching_features(self, i, current_feats, lookup_feats, poses, K, invK, min_depth_bin=None, max_depth_bin=None):

        if self.adaptive_bins:
            self.compute_depth_bins(min_depth_bin, max_depth_bin)

        batch_size, chns, height, width = current_feats.shape
        lookup_feats = lookup_feats.reshape(batch_size, lookup_feats.shape[0]//batch_size, chns, height, width)

        # warp features to find cost volume
        cost_volume, missing_mask = \
            self.match_features(current_feats, lookup_feats, poses, K, invK)
        confidence_mask = self.compute_confidence_mask(cost_volume.detach() *
                                                        (1 - missing_mask.detach()))
    
        # for visualisation - ignore 0s in cost volume for minimum
        viz_cost_vol = cost_volume.clone().detach()
        viz_cost_vol[viz_cost_vol == 0] = 100
        mins, argmin = torch.min(viz_cost_vol, 1)
        lowest_cost = self.indices_to_disparity(argmin)
        
        # mask the cost volume based on the confidence
        cost_volume *= confidence_mask.unsqueeze(1)
        post_matching_feats = self.reduce_convs[i](torch.cat([current_feats, cost_volume], 1)) + current_feats

        return post_matching_feats, lowest_cost, confidence_mask
    
    def _fuse_matching_cross_attention(self, current_feats, lookup_feats):
        batch_size, channels, height, width = current_feats.shape
        
        # Reshape current_feats
        current_feats_flat = current_feats.view(batch_size, channels, -1).permute(2, 0, 1)
        
        # Reshape lookup_feats
        lookup_feats_flat = lookup_feats.view(batch_size, -1, height * width).permute(2, 0, 1)
        
        # Pass through transformer decoder layers
        output = current_feats_flat
        for layer in self.transformer_layers:
            output = layer(output, lookup_feats_flat)

        # Reshape output back to original dimensions
        fused_features = output.permute(1, 2, 0).view(batch_size, channels, height, width)

        return fused_features
    
    def _fuse_features_preciver(self, current_feats, lookup_feats):
        batch_size, channels, height, width = current_feats.shape
        
        # Reshape current_feats
        current_feats_flat = current_feats.view(batch_size, channels, -1)
        
        # Reshape lookup_feats
        lookup_feats_flat = lookup_feats.view(batch_size, -1, height * width)
        
        #concatenate features along channel dim
        fused_features = torch.cat((current_feats_flat, lookup_feats_flat), 1).permute(0,2,1)
        output = self.preciver(fused_features).permute(0,2,1)
        output = output.view(batch_size, channels, height, width)

        return output

    def forward(self, out_features, lookup_features, patch_h, patch_w,  poses, K, invK,
                min_depth_bin=None, max_depth_bin=None):
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
            if i < 3:
                _, lowest_cost, confidence_mask = self._fuse_matching_features(i, x, lookup_feature, poses, K, invK, min_depth_bin, max_depth_bin)
            
                x = self._fuse_features_preciver(x, lookup_feature)
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
        return depth, lowest_cost, confidence_mask, out_

