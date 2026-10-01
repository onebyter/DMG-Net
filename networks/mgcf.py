import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

class AdaptiveGranularContextAggregation(nn.Module):
    def __init__(self, output_dim, granularity_size):
        super().__init__()
        self.output_dim = output_dim
        self.granularity_size = granularity_size
        self.mlp1 = nn.Linear(granularity_size * granularity_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.context_query_vector = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.feature_refinement_matrix = torch.nn.parameter.Parameter(torch.eye(output_dim),
                                                                       requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        G_s = self.granularity_size

        regional_patches = x.unfold(1, G_s, G_s).unfold(2, G_s, G_s)
        regional_patches = regional_patches.reshape(B, -1, G_s * G_s, C)
        regional_patches = regional_patches.mean(dim=-1)

        regional_patches = self.mlp1(regional_patches)
        regional_patches = self.norm(regional_patches)
        regional_patches = self.mlp2(regional_patches)

        local_attention = F.softmax(regional_patches, dim=-1)
        context_aware_features = regional_patches * local_attention

        cos_sim = F.normalize(context_aware_features, dim=-1) @ F.normalize(self.context_query_vector[None, ..., None], dim=1)
        mask = cos_sim.clamp(0, 1)
        context_aware_features = context_aware_features * mask
        refined_features = context_aware_features @ self.feature_refinement_matrix

        refined_features = refined_features.reshape(B, H // G_s, W // G_s, self.output_dim)
        refined_features = refined_features.permute(0, 3, 1, 2)
        output = F.interpolate(refined_features, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(output)

        return output

class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x

class ContextRefinementBlock(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

class MGCF(nn.Module):
    def __init__(self, in_features, filters) -> None:
        super().__init__()

        self.skip = ContextRefinementBlock(in_features=in_features,
                                           out_features=filters,
                                           kernel_size=(1, 1),
                                           padding=(0, 0),
                                           norm_type='bn',
                                           activation=False)
        self.c1 = ContextRefinementBlock(in_features=in_features,
                                         out_features=filters,
                                         kernel_size=(3, 3),
                                         padding=(1, 1),
                                         norm_type='bn',
                                         activation=True)
        self.c2 = ContextRefinementBlock(in_features=filters,
                                         out_features=filters,
                                         kernel_size=(3, 3),
                                         padding=(1, 1),
                                         norm_type='bn',
                                         activation=True)
        self.c3 = ContextRefinementBlock(in_features=filters,
                                         out_features=filters,
                                         kernel_size=(3, 3),
                                         padding=(1, 1),
                                         norm_type='bn',
                                         activation=True)
        self.sa = SpatialAttentionModule()
        self.cn = ECA(filters)
        self.agca2 = AdaptiveGranularContextAggregation(filters, 2)
        self.agca4 = AdaptiveGranularContextAggregation(filters, 4)

        self.bn1 = nn.BatchNorm2d(filters)
        self.drop = nn.Dropout2d(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_agca2 = self.agca2(x_skip)
        x_agca4 = self.agca4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)

        x = x1 + x2 + x3 + x_skip + x_agca2 + x_agca4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
