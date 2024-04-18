import torch
import torch.nn as nn
import math

from timm.models.layers import trunc_normal_
from model.encoder.pvt import pvt_v2_b5
from model.encoder.swin import SwinTransformer
from model.decoder.convguideddecoder import ConvGuidedDecoder
from model.head.mrf_head import MRFHead, SampleHead

class ConvGuidedDepth(nn.Module):
    def __init__(self, pretrained=None, frozen_stages=-1, min_depth=0.1, max_depth=100.0, **kwargs):
        super(ConvGuidedDepth, self).__init__()

        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
        use_sample_head = False

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=7,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )

        self.encoder = SwinTransformer(**backbone_cfg)
        self.decoder = ConvGuidedDecoder()

        if use_sample_head is True:
            self.head = SampleHead(max_depth=max_depth)
        else:
            self.head = MRFHead(max_depth=max_depth, min_depth=min_depth)
        

        self.init_weight(pretrained=pretrained)

    def forward(self, x):
        en_feats = self.encoder(x)
        de_feats = self.decoder(en_feats)
        depth = self.head(de_feats)
        return depth

    def init_weight(self, pretrained=None):
        self.encoder.init_weights(pretrained=pretrained)
        self.decoder.init_weight()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  
            nn.init.constant_(m.weight, 1.0)  
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

if __name__ == '__main__':
    inputs = torch.randn((1, 3, 640, 480))
    m = ConvGuidedDepth()
    out = m(inputs)
    print(out.shape)
