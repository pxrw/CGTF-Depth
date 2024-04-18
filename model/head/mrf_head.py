import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.utils import up_sampling_by_interpolate

class MRFHead(nn.Module):
    def __init__(self, max_depth, min_depth, in_features=512, out_features=256, act_layer=nn.GELU()):
        super(MRFHead, self).__init__()
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.act_layer = act_layer
        hidden_features = in_features * 4
        out_features = out_features if out_features is not None else in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.conv = nn.Conv2d(in_channels=512, out_channels=out_features, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=1)
        self.upsample = up_sampling_by_interpolate

    def forward(self, feats):
        process_feat4, feat4, feat3, feat2, feat1 = feats

        x = F.avg_pool2d(process_feat4, 1).flatten(start_dim=2)
        x = torch.mean(x, dim=2)
        x = self.act_layer(self.fc1(x))
        x = self.fc2(x)
        bins = torch.softmax(x, dim=1)
        bins = bins / bins.sum(dim=1, keepdim=True)
        bin_widths = (self.max_depth - self.min_depth) * bins
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.contiguous().view(n, dout, 1, 1) # 512

        feat1 = self.conv(feat1)
        depth_weight = self.softmax(feat1)
        depth = torch.sum(depth_weight * centers, dim=1, keepdim=True)
        return self.upsample(depth, 4)
    
class SampleHead(nn.Module):
    def __init__(self, max_depth, in_features=512):
        super(SampleHead, self).__init__()
        self.upsample = up_sampling_by_interpolate
        self.max_depth = max_depth
        hidden_feat = in_features // 4
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=hidden_feat, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_feat, out_channels=1, kernel_size=1)

    def forward(self, feats):
        process_feat4, feat4, feat3, feat2, feat1 = feats
        feat1 = self.conv2(self.conv1(feat1))
        depth = self.upsample(feat1, 4)
        depth = torch.sigmoid(depth)
        return depth * self.max_depth