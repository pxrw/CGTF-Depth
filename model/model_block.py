import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_, to_2tuple

from model.encoder.swin import BasicLayer
from model.encoder.swin import window_partition, window_reverse, Mlp
from model.dcg import DCNv3

class CBAMLayer(nn.Module):
    def __init__(self, in_channels, reductions=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reductions, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels // reductions, in_channels, kernel_size=1, bias=False)
        )

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=spatial_kernel, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = x * channel_out
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = x * spatial_out
        return x

class CBRBlock(nn.Module):
    '''
    一个标准的conv + bn + relu的模块
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, use_dcn=False):
        super(CBRBlock, self).__init__()
        if use_dcn:
            self.conv = DCNv3(in_channels)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ConvGuidedLayer(nn.Module):
    '''
    一个普通的卷积模块（conv + bn + relu）后面跟了一个CBAM注意力
    '''
    def __init__(self, in_channels, kernel_size=3, stride=1, use_dcn=False):
        super(ConvGuidedLayer, self).__init__()
        self.cbr = CBRBlock(in_channels, in_channels, kernel_size, stride, padding=1, use_dcn=use_dcn)
        self.channel_spatial_attn = CBAMLayer(in_channels)

    def forward(self, x):
        res = x
        x = self.cbr(x)
        x = self.channel_spatial_attn(x)
        # x = F.relu(x)
        out = x + res
        return out

class WindowTransExt(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, window_size=7, norm_layer=nn.LayerNorm):
        super(WindowTransExt, self).__init__()
        if in_channels != embed_dim:
            self.proj = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        else:
            self.proj = None

        self.block = BasicLayer(
            dim=embed_dim,
            depth=1,
            num_heads=num_heads,
            window_size=window_size
        )
        layer = norm_layer(embed_dim)
        layer_name = 'norm_window_trans'
        self.add_module(layer_name, layer)

    def forward(self, x):
        b, c, h, w = x.shape
        if self.proj is not None:
            x = self.proj(x)
        shortcut_x = x
        x = x.flatten(2).transpose(1, 2)
        out = self.block(x, h, w)[0]
        out = out.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return out + shortcut_x

class FusionAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, v_dim, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(v_dim, v_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, v, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        q = self.q(x).view(B_, N, self.num_heads, -1).transpose(1, 2)

        kv = self.kv(v).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FusionBlock(nn.Module):
    def __init__(self, dim, num_heads, v_dim, window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0.):
        super(FusionBlock, self).__init__()
        self.window_size = window_size
        self.dim = dim
        self.num_heads = num_heads
        self.v_dim = v_dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        act_layer = nn.GELU
        norm_layer = nn.LayerNorm

        self.norm1 = norm_layer(dim)
        self.normv = norm_layer(dim)
        self.attn = FusionAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, v_dim=v_dim,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(v_dim)
        mlp_hidden_dim = int(v_dim * mlp_ratio)
        self.mlp = Mlp(in_features=v_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, v, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        shortcut_v = v
        v = self.normv(v)
        v = v.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        v = F.pad(v, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # partition windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        v_windows = window_partition(v, self.window_size)  # nW*B, window_size, window_size, C
        v_windows = v_windows.view(-1, self.window_size * self.window_size,
                                   v_windows.shape[-1])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, v_windows, mask=None)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.v_dim)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, self.v_dim)

        # FFN
        x = self.drop_path(x) + shortcut
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, H, W

class ConvTransFusion(nn.Module):
    def __init__(self, input_dim, embed_dim, v_dim, window_size=7, num_heads=4, norm_layer=nn.LayerNorm):
        super(ConvTransFusion, self).__init__()
        self.embed_dim = embed_dim
        if input_dim != embed_dim:
            self.proj_conv = nn.Conv2d(input_dim, embed_dim, 3, padding=1)
        else:
            self.proj_conv = None

        if v_dim != embed_dim:
            self.proj_trans = nn.Conv2d(v_dim, embed_dim, 3, padding=1)
        elif embed_dim % v_dim == 0:
            self.proj_trans = None

        v_dim = embed_dim
        self.fusion_block = FusionBlock(
            dim=embed_dim,
            num_heads=num_heads,
            v_dim=v_dim,
            window_size=window_size,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
        )

        layer = norm_layer(embed_dim)
        layer_name = 'norm_fusion'
        self.add_module(layer_name, layer)

    def forward(self, c, t):
        if self.proj_conv is not None:
            c = self.proj_conv(c)

        if self.proj_trans is not None:
            t = self.proj_trans(t)

        shortcut_c = c
        shortcut_t = t

        Wh, Ww = c.size(2), c.size(3)
        c = c.flatten(2).transpose(1, 2)
        t = t.flatten(2).transpose(1, 2)

        out, H, W = self.fusion_block(c, t, Wh, Ww)
        norm_layer = getattr(self, f'norm_fusion')
        out = norm_layer(out)
        out = out.view(-1, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        return out + shortcut_c + shortcut_t

class ConvGuidedTransFusionBlockV1(nn.Module):
    def __init__(self, in_channels, num_heads, embed_dim, sr_ratio=1):
        super(ConvGuidedTransFusionBlockV1, self).__init__()
        self.conv_guided_layer = ConvGuidedLayer(in_channels)
        self.global_feat_ext = WindowTransExt(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        self.conv_trans_fusion = ConvTransFusion(
            input_dim=in_channels,
            embed_dim=embed_dim,
            v_dim=embed_dim,
            # sr_ratio=sr_ratio
        )

    def forward(self, x):
        shortcut_x = x
        conv_stream = self.conv_guided_layer(x)
        trans_stream = self.global_feat_ext(x)
        fusion_out = self.conv_trans_fusion(conv_stream, trans_stream)
        return fusion_out + shortcut_x

class ConvGuidedTransFusionBlockV2(nn.Module):
    def __init__(self, in_channels, num_heads, embed_dim, use_dcn=False):
        super(ConvGuidedTransFusionBlockV2, self).__init__()
        self.conv_guided_layer = ConvGuidedLayer(in_channels, use_dcn=use_dcn)

        self.conv_trans_fusion = ConvTransFusion(
            input_dim=in_channels,
            embed_dim=embed_dim,
            v_dim=in_channels,
            num_heads=num_heads,
        )

        if in_channels != embed_dim:
            self.proj_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        else:
            self.proj_conv = None

    def forward(self, x):
        if self.proj_conv is not None:
            shortcut_x = self.proj_conv(x)
        else:
            shortcut_x = x
        conv_stream = self.conv_guided_layer(x)
        fusion_out = self.conv_trans_fusion(conv_stream, x)
        return fusion_out + shortcut_x

##################################################################################################################
class SRAttention(nn.Module):
    def __init__(self, dim, v_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(SRAttention, self).__init__()
        self.dim = dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.sr_ratio = sr_ratio
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(v_dim, v_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, v, h, w):
        B, N, C = x.shape
        # [b, n, c] -> [b, n, head, c // head] -> [b, head, n, c // head]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            v_ = v.permute(0, 2, 1).reshape(B, C, h, w)   # [b, n, c] -> [b, c, n] -> [b, c, h, w]
            v_ = self.sr(v_).reshape(B, C, -1).permute(0, 2, 1)  # [b, c, h', w'] -> [b, c, n'] -> [b, n', c]
            v_ = self.norm(v_)
            # [b, n', c] -> [b, n', 2c] -> [b, n', 2, head, c // head] -> [2, b, head, n', c // head]
            kv = self.kv(v_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(v).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv.unbind(0)  # [b, head, n', c // head]

        # [b, head, n, c // head] @ [b, head, c // head, n'] -> [b, head, n, n']
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # [b, head, n, n'] @ [b, head, n', c // head] -> [b, head, n, c // head] -> [b, n, head, c // head] -> [b, n, c]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SRFusionBlock(nn.Module):
    def __init__(self, in_channel, v_dim, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(SRFusionBlock, self).__init__()
        self.embed_dim = embed_dim
        if in_channel != embed_dim:
            self.proj_conv = nn.Conv2d(in_channel, embed_dim, 3, padding=1)
        else:
            self.proj_conv = None

        if v_dim != embed_dim:
            self.proj_trans = nn.Conv2d(v_dim, embed_dim, 3, padding=1)
        else:
            self.proj_trans = None

        v_dim = embed_dim
        self.sr_attn = SRAttention(dim=embed_dim, v_dim=v_dim, num_heads=num_heads, sr_ratio=sr_ratio)
        layer = norm_layer(embed_dim)
        layer_name = 'norm_fusion'
        self.add_module(layer_name, layer)

    def forward(self, c, t):
        if self.proj_conv is not None:
            c = self.proj_conv(c)
        if self.proj_trans is not None:
            t = self.proj_trans(t)

        shortcut_c = c
        shortcut_t = t

        h, w = c.size(2), c.size(3)
        c = c.flatten(2).transpose(1, 2)
        t = t.flatten(2).transpose(1, 2)
        out = self.sr_attn(c, t, h, w)
        norm_layer = getattr(self, f'norm_fusion')
        out = norm_layer(out)
        out = out.view(-1, h, w, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        return out + shortcut_c + shortcut_t

class SRConvGuidedTransFusionBlock(nn.Module):
    def __init__(self, in_channels, num_heads, embed_dim, sr_ratio=1, use_dcn=False):
        super(SRConvGuidedTransFusionBlock, self).__init__()
        self.conv_guided_layer = ConvGuidedLayer(in_channels, use_dcn=use_dcn)

        self.srconv_trans_fusion = SRFusionBlock(
            in_channel=in_channels,
            embed_dim=embed_dim,
            v_dim=in_channels,
            num_heads=num_heads,
            sr_ratio=sr_ratio
        )

    def forward(self, x):
        conv_stream = self.conv_guided_layer(x)
        fusion_out = self.srconv_trans_fusion(conv_stream, x)
        return fusion_out


if __name__ == '__main__':
    input1 = torch.randn((1, 96, 20, 10))
    input2 = torch.randn((1, 64, 20, 10))

    m = SRConvGuidedTransFusionBlock(in_channels=96, num_heads=4, embed_dim=64, sr_ratio=4)
    out = m(input1)
    print(out.shape)