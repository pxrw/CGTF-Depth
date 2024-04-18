import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.nn.init import constant_, xavier_uniform_


def _is_power_of_2(n):
    '''
    用来判断是否是2的n次方，且n不为0，如果n & (n - 1) = 0 说明n是2的某次方
    '''
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class to_channels_first(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):  # [b, h, w, c] -> [b, c, h, w]
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):  # [b, c, h, w] -> [b, h, w, c]
        return x.permute(0, 2, 3, 1)


class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,  # (group,channels)
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()  # 全连接层+sigmod
        # F.linear的输出取决于传入的实参weight和bias
        # 输入:(*,channels) -> 输出：(*,group)
        return center_feature_scale


def build_norm_layer(dim, norm_layer, in_format='channels_last',
                     out_format='channels_last', eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())  # (b,c,h,w)->(b,h,w,c)
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


def _get_reference_points(spatial_shapes, device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h=0, pad_w=0,
                          stride_h=1, stride_w=1):
    # spatial_shapes: 原始输入pad后大小 (N_,H_in,W_in,C)
    _, H_, W_, _ = spatial_shapes
    # 等于offset的宽和高(pad=1,stride=1)
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1  # H_out
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1  # W_out

    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out,
            dtype=torch.float32,
            device=device),
        # 在[pad_h + 0.5,H_ - pad_h - 0.5]范围内，生成H_out个等分点
        torch.linspace(
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
            dtype=torch.float32,
            device=device))
    # 在[pad_w + 0.5,W_ - pad_w - 0.5]范围内，生成W_out个等分点
    # ref_y:(H_out,W_out)
    # ref_x:(H_out,W_out)
    ref_y = ref_y.reshape(-1)[None] / H_  # 归一化
    # reshape后(H_out*W_out)
    # None后(1,H_out*W_out)
    ref_x = ref_x.reshape(-1)[None] / W_
    # (1,H_out*W_out)

    ref = torch.stack((ref_x, ref_y), -1).reshape(
        1, H_out, W_out, 1, 2)
    # stack后 (1,H_out*W_out,2)
    # reshape 后 (1,H_out,W_out,1,2)

    return ref  # (1,H_out_cov,W_out_cov,1,2)


def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w, dilation_h, dilation_w, group, device):
    _, H_, W_, _ = spatial_shapes  # (N_,H_in,W_in,C)
    points_list = []
    x, y = torch.meshgrid(
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) +
            (kernel_w - 1) * dilation_w, kernel_w,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) +
            (kernel_h - 1) * dilation_h, kernel_h,
            dtype=torch.float32,
            device=device))
    # x: (kernel_w,kernel_h)
    # y: (kernel_w,kernel_h)
    points_list.extend([x / W_, y / H_])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2). \
        repeat(1, group, 1).permute(1, 0, 2)
    # stack后：(kernel_w,kernel_h,2)
    # reshape后：(kernel_w*kernel_h,1,2)
    # repeat后：(kernel_w*kernel_h,group,2)
    # permute后：(group,kernel_w*kernel_h,2)
    grid = grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)
    # (1,1,1,group * kernel_h * kernel_w,2)
    return grid


def dcnv3_core_pytorch(
        input, offset, mask, kernel_h,
        kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w, group,
        group_channels, offset_scale):
    # for debug and test only,
    # need to use cuda version instead
    '''
    输入参数说明：
    input: (N,H,W,C)
    offset: (N,H,W,group*2*kernel_size*kernel_size)
    mask: (N,H,W,group*2*kernel_size*kernel_size)
    pad_h=pad_w 在h,w方向的填充（默认是1）
    dilation_h,dilation_w : 空洞卷积率
    group：分组数
    group_channels: 每个组中的通道数
    '''
    input = F.pad(
        input,
        [0, 0, pad_h, pad_h, pad_w, pad_w])  # 用0填充 大小变为 (N_,H_in,W_in,C)
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape  # (N_,H_out,W_out,C)=(N,H,W,C)

    ref = _get_reference_points(
        input.shape, input.device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h, pad_w, stride_h, stride_w)
    # ref (1,H_out_cov,W_out_cov,1,2)
    grid = _generate_dilation_grids(
        input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.device)
    # grid (1,1,1,group * kernel_h * kernel_w,2)
    spatial_norm = torch.tensor([W_in, H_in]).reshape(1, 1, 1, 2). \
        repeat(1, 1, 1, group * kernel_h * kernel_w).to(input.device)
    # spatial_norm (1,1,1,group*kernel_h*kernel_w)
    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1).flatten(3, 4) + \
                         offset * offset_scale / spatial_norm  # 得到最终的采样点位置
    # repeat 后(N_,H_out,W_out,group*kernel_h*kernel_w,2)
    # flatten 后(N_,H_out,W_out,group*kernel_h*kernel_w*2)
    # offset: (N_,H_out,W_out,group*kernel_h*kernel_w*2)
    P_ = kernel_h * kernel_w
    sampling_grids = 2 * sampling_locations - 1  # 把大小规范化到[-1,1]之间,方便F.grid_sample函数进行采样

    input_ = input.view(N_, H_in * W_in, group * group_channels).transpose(1, 2). \
        reshape(N_ * group, group_channels, H_in, W_in)
    # input: (N_,H_in,W_in,group*group_channels)
    # view后:(N_, H_in*W_in, group*group_channels)
    # transpose后：(N_, group*group_channels, H_in*W_in)
    # reshape后：(N_*group, group_channels, H_in, W_in)

    sampling_grid_ = sampling_grids.view(N_, H_out * W_out, group, P_, 2).transpose(1, 2). \
        flatten(0, 1)
    # sampling_grids:(N_, H_out, W_out, group*P_*2)
    # transpose后：(N_, H_out*W_out, group, P_, 2)
    # flatten后：( N_*group, H_out*W_out, P_, 2)

    sampling_input_ = F.grid_sample(  # 偏移后的采样位置
        input_, sampling_grid_, mode='bilinear', padding_mode='zeros', align_corners=False)  # 双线性插值
    # input_: (N_*group, group_channels, H_in, W_in)
    # sampling_grid_: (N_*group, H_out*W_out, P_, 2)
    # sampling_input_: (N_*group, group_channels, H_out*W_out, P_)

    mask = mask.view(N_, H_out * W_out, group, P_).transpose(1, 2). \
        reshape(N_ * group, 1, H_out * W_out, P_)  # 调制标量
    # mask: (N_, H_out, W_out, group*P_)
    # view后: (N_, H_out*W_out, group, P_)
    # transpose后：(N_, group, H_out*W_out, P_)
    # reshape后：(N_*group, 1, H_out*W_out, P_)

    output = (sampling_input_ * mask).sum(-1).view(N_, group * group_channels, H_out * W_out)
    # *后：(N_*group,  group_channels, H_out*W_out, P_)
    # sum(-1)后：(N_*group,  group_channels, H_out*W_out)
    # view后：(N_,group*group_channels, H_out*W_out)

    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous()
    # transpose后：(N_, H_out*W_out，group*group_channels)
    # reshape后：(N_, H_out, W_out,group*group_channels)=(N,H,W,C)


class DCNv3(nn.Module):
    def __init__(self, channels, kernel_size=3, dw_kernel_size=None,
                 stride=1, pad=1, dilation=1, group=4, offset_scale=1.0,
                 act_layer='GELU', norm_layer='LN', center_feature_scale=False):
        super().__init__()
        if channels % group != 0:  # 分组卷积必须保证通道数可以被组数整除
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group  # 每个组拥有的通道数
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        if not _is_power_of_2(_d_per_group):  # 最好将每个组拥有的通道数设置为2的某次方，这样方便cuda具体实现
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.center_feature_scale = center_feature_scale

        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=dw_kernel_size, stride=1,
                      padding=(dw_kernel_size - 1) // 2, groups=channels),
            build_norm_layer(channels, norm_layer, 'channels_first', 'channels_last'),
            build_act_layer(act_layer)
        )

        self.offset = nn.Linear(channels, group * kernel_size * kernel_size * 2)
        self.mask = nn.Linear(channels, group * kernel_size * kernel_size)

        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()

        if center_feature_scale:  # 如果需要对特征图进行缩放
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))  # weighit:大小为(group,channels)的全0 Tensor
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))  # bias: 大小为(group)的全0 tensor
            self.center_feature_scale_module = CenterFeatureScaleModule()  # (*,channels)->(*,groups)

    # 参数初始化
    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):  # (N,H,W,C)
        # print('use dcn conv...')
        N, C, H, W = input.shape
        input = input.permute(0, 2, 3, 1)
        x = self.input_proj(input)  # Linear层，通道不变 x: (N,H,W,C)
        x_proj = x
        x1 = input.permute(0, 3, 1, 2)  # x1: (N,C,H,W)
        '''
        改进点1：将原始卷积权值w_k分离为depth-wise和point-wise两部分，其中depth-wise部分由原始的location-aware modulation scalar m_k负责，point-wise部分是采样点之间的共享投影权值w。
        '''
        # DW_conv(逐通道卷积)+Norm+Act
        x1 = self.dw_conv(x1)  # (N,H,W,C)
        ''' 
        改进点2：我们将空间聚集过程分成G组，每个组都有单独的采样偏移offset和调制规模mask
        因此在一个卷积层上的不同组可以有不同的空间聚集模式，从而为下游任务带来更强的特征。
        '''
        offset = self.offset(x1)  # 生成偏移 (N,H,W,group*2*kernel_size*kernel_size)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)  # mask 表示调制标量
        # mask后：(N,H,W,group*kernel_size*kernel_size)
        # reshape后：(N,H,W,group,kernel_size*kernel_size)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)  # (N,H,W,group*kernel_size*kernel_size)
        # softmax的dim=-1,表示在kernel_size*kernel_size个样本点进行归一化，其和等于1。
        '''
        改进点3：我们将基于element-wise的sigmoid归一化改为基于样本点的softmax归一化。这样，将调制标量的和限制为1，使得不同尺度下模型的训练过程更加稳定。
        '''
        x = dcnv3_core_pytorch(  # 可变形卷积的核心代码
            x, offset, mask,
            # x: (N,H,W,C)
            # offset: (N,H,W,group*2*kernel_size*kernel_size)
            # mask: (N,H,W,group*kernel_size*kernel_size)
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale)  # (N_, H_out, W_out,group*group_channels)

        if self.center_feature_scale:  # 如果需要特征缩放
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # x1：(N,H,W,C) ->(N,H,W,groups)
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            #  (N, H, W, groups) -> (N, H, W, groups, 1) -> (N, H, W, groups, _d_per_group) -> (N, H, W, channels)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
            # x_proj和x按照一定的系数相加
        x = self.output_proj(x)  # (N,H,W,C)
        return x.permute(0, 3, 1, 2)