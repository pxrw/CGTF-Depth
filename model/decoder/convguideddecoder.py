import torch
import torch.nn as nn

from utils.utils import up_sampling_by_convt
from model.model_block import ConvGuidedTransFusionBlockV2, SRConvGuidedTransFusionBlock
from model.psp import PSP


class ConvGuidedDecoder(nn.Module):
    def __init__(self):
        super(ConvGuidedDecoder, self).__init__()

        use_sr = False
        use_dcn = False
        embed_dim = 512
        embed_dims = [64, 128, 256, embed_dim]
        num_heads = [32, 16, 8, 4]
        in_channels = [192, 384, 768, 1536]
        hidden_channels = [128, 256, 512]
        sr_ratios = [1, 2, 4, 8] if use_sr is True else None

        norm_cfg = dict(type='BN', requires_grad=True)
        process_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        self.preprocess = PSP(**process_cfg)

        if use_sr:
            self.stage4 = SRConvGuidedTransFusionBlock(
                in_channels=embed_dim,
                num_heads=num_heads[0],
                embed_dim=embed_dims[0],
                sr_ratio=sr_ratios[0],
                use_dcn=use_dcn
            )

            self.stage3 = SRConvGuidedTransFusionBlock(
                in_channels=hidden_channels[0],
                num_heads=num_heads[1],
                embed_dim=embed_dims[1],
                sr_ratio=sr_ratios[1],
                use_dcn=use_dcn
            )

            self.stage2 = SRConvGuidedTransFusionBlock(
                in_channels=hidden_channels[1],
                num_heads=num_heads[2],
                embed_dim=embed_dims[2],
                sr_ratio=sr_ratios[2],
                use_dcn=use_dcn
            )

            self.stage1 = SRConvGuidedTransFusionBlock(
                in_channels=hidden_channels[2],
                num_heads=num_heads[3],
                embed_dim=embed_dims[3],
                sr_ratio=sr_ratios[3],
                use_dcn=use_dcn
            )

        else:
            self.stage4 = ConvGuidedTransFusionBlockV2(in_channels=embed_dim,
                                                       num_heads=num_heads[0],
                                                       embed_dim=embed_dims[0],
                                                       use_dcn=use_dcn)
            self.stage3 = ConvGuidedTransFusionBlockV2(in_channels=hidden_channels[0],
                                                       num_heads=num_heads[1],
                                                       embed_dim=embed_dims[1],
                                                       use_dcn=use_dcn)
            self.stage2 = ConvGuidedTransFusionBlockV2(in_channels=hidden_channels[1],
                                                       num_heads=num_heads[2],
                                                       embed_dim=embed_dims[2],
                                                       use_dcn=use_dcn)
            self.stage1 = ConvGuidedTransFusionBlockV2(in_channels=hidden_channels[2],
                                                       num_heads=num_heads[3],
                                                       embed_dim=embed_dims[3],
                                                       use_dcn=use_dcn)

        self.up4 = up_sampling_by_convt(in_channels=embed_dims[0], out_channels=embed_dims[0] // 2)
        self.up3 = up_sampling_by_convt(in_channels=embed_dims[1], out_channels=embed_dims[1] // 2)
        self.up2 = up_sampling_by_convt(in_channels=embed_dims[2], out_channels=embed_dims[2] // 2)

        self.conv3 = nn.Conv2d(embed_dims[0] // 2 + in_channels[2], hidden_channels[0], kernel_size=1)
        self.conv2 = nn.Conv2d(embed_dims[1] // 2 + in_channels[1], hidden_channels[1], kernel_size=1)
        self.conv1 = nn.Conv2d(embed_dims[2] // 2 + in_channels[0], hidden_channels[2], kernel_size=1)

    def forward(self, feats):
        feat1, feat2, feat3, feat4 = feats
        process_feat4 = self.preprocess(feats)
        feat4 = self.stage4(process_feat4)  # 128

        feat4_up = self.up4(feat4)  # 64
        feat3 = torch.cat([feat4_up, feat3], dim=1)
        feat3 = self.conv3(feat3)
        feat3 = self.stage3(feat3)

        feat3_up = self.up3(feat3)
        feat2 = torch.cat([feat3_up, feat2], dim=1)
        feat2 = self.conv2(feat2)
        feat2 = self.stage2(feat2)

        feat2_up = self.up2(feat2)
        feat1 = torch.cat([feat2_up, feat1], dim=1)
        feat1 = self.conv1(feat1)
        feat1 = self.stage1(feat1)
        return process_feat4, feat4, feat3, feat2, feat1

    def init_weight(self):
        pass


if __name__ == '__main__':
    in_channels = [192, 384, 768, 1536]
    input1 = torch.randn((1, 192, 320, 240))
    input2 = torch.randn((1, 384, 160, 120))
    input3 = torch.randn((1, 768, 80, 60))
    input4 = torch.randn((1, 1536, 40, 30))

    m = ConvGuidedDecoder()
    outs = m([input1, input2, input3, input4])
    for out in outs:
        print(out.shape)
