# -*- coding: utf-8 -*-
#  @Author: KunchangLi
#  @Date: 2020-02-10 20:01:23
#  @LastEditor: KunchangLi
#  @LastEditTime: 2020-03-26 20:48:53

from math import log2, pow, ceil, floor

import torch
import torch.nn as nn
import torch.nn.functional as F

from arch import ResNet

__all__ = ['make_reshape_block']


def conv_3x1x1_bn(inp, oup, groups=1):
    return nn.Sequential(
        nn.Conv3d(inp, oup, (3, 1, 1), (1, 1, 1), (1, 0, 0), groups=groups, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x3x3_bn(inp, oup, groups=1, identity=False):
    if identity:
        return nn.Sequential(
            nn.Conv3d(inp, oup, (1, 3, 3), (1, 1, 1), (0, 1, 1), groups=groups, bias=False),
        nn.BatchNorm3d(oup),
        )
    else:
        return nn.Sequential(
            nn.Conv3d(inp, oup, (1, 3, 3), (1, 1, 1), (0, 1, 1), groups=groups, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
        )


def conv_1x3x3(inp, oup, groups=1):
    return nn.Sequential(
        nn.Conv3d(inp, oup, (1, 3, 3), (1, 1, 1), (0, 1, 1), groups=groups, bias=False),
    )


def conv_3x3_bn(inp, oup, groups=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, 1, 1, groups=groups, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup, groups=1, identity=False):
    if identity:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, groups=groups, bias=False),
            nn.BatchNorm2d(oup)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, groups=groups, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )


def conv_1x1x1_bn(inp, oup, groups=1, identity=False):
    if identity:
        return nn.Sequential(
            nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups, bias=False),
            nn.BatchNorm3d(oup),
        )
    else:
        return nn.Sequential(
            nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups, bias=False),
            nn.BatchNorm3d(oup),
            nn.ReLU(inplace=True)
        )


def conv_1x1x1(inp, oup, groups=1):
    return nn.Sequential(
        nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups, bias=False),
    )


def conv_3x1x1_bn(inp, oup, groups=1, identity=False):
    if identity:
        return nn.Sequential(
            nn.Conv3d(inp, oup, (3, 1, 1), (1, 1, 1), (1, 0, 0), groups=groups, bias=False),
            nn.BatchNorm3d(oup),
        )
    else:
        return nn.Sequential(
            nn.Conv3d(inp, oup, (3, 1, 1), (1, 1, 1), (1, 0, 0), groups=groups, bias=False),
            nn.BatchNorm3d(oup),
            nn.ReLU(inplace=True)
        )


def conv_3x1x1(inp, oup, groups=1):
    return nn.Sequential(
        nn.Conv3d(inp, oup, (3, 1, 1), (1, 1, 1), (1, 0, 0), groups=groups, bias=False),
    )


def check_replace(num_total, i):
    res = False
    if num_total == 4:
        if i == 1:
            res = True
    elif num_total == 7:
        if i % 2 == 1:
            res = True
    elif num_total == 12:
        if i > 0:
            res = True
    elif num_total == 3:
        if i % 2 == 1 and i != 1:
            res = True
    elif num_total == 5:
        if i > 0 and i % 2 != 1:
            res = True
    return res


def make_reshape_block(net, num_total, diff_div=8, num_segments=8, aggregation='sum'):
    if isinstance(net, ResNet):
        def _deal_block(stage):
            blocks = list(stage.children())
            for i, b in enumerate(blocks):
                if check_replace(num_total, i):
                    inplanes = b.inplanes
                    planes = b.planes
                    downsample = b.downsample
                    reshape_block = ReshapeBlock(inplanes, planes, downsample, 
                    num_segments, diff_div, aggregation)
                    reshape_block.conv1[0] = blocks[i].conv1
                    reshape_block.conv1[1] = blocks[i].bn1
                    for p in reshape_block.conv1.parameters():
                        if not p.requires_grad:
                            p.requires_grad = True
                    if aggregation == 'sum':
                        reshape_block.conv3[0] = blocks[i].conv3
                        reshape_block.conv3[1] = blocks[i].bn3
                        for p in reshape_block.conv3.parameters():
                            if not p.requires_grad:
                                p.requires_grad = True
                    blocks[i] = reshape_block
            return nn.Sequential(*blocks)

        net.layer1 = _deal_block(net.layer1)
        net.layer2 = _deal_block(net.layer2)
        net.layer3 = _deal_block(net.layer3)
        net.layer4 = _deal_block(net.layer4)
    else:
        raise NotImplementedError


class ReshapeBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, downsample=None, num_segments=8, diff_div=8, aggregation='sum'):
        super(ReshapeBlock, self).__init__()
        self.downsample = downsample
        self.aggregation = aggregation
        print("parallel, spatiotemporal plus")
        self.conv1 = conv_1x1_bn(inplanes, planes)
        self.conv2 = ReshapeInteraction(planes, num_segments=num_segments)
        self.conv3 = conv_1x1_bn(planes, planes * self.expansion, identity=True)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


def get_xy(total_num):
    temp = log2(total_num) / 2.0
    x = int(pow(2, floor(temp)))
    y = int(pow(2, ceil(temp)))
    return x, y


class ReshapeInteraction(nn.Module):
    def __init__(self, planes, num_segments=8):
        super(ReshapeInteraction, self).__init__()
        assert planes % num_segments == 0, 'planes should be divisible by num_segments'
        self.t = num_segments
        self.x, self.y = get_xy(planes)
        self.conv1_x = conv_1x3x3_bn(planes, planes, groups=self.x)
        self.conv1_z = conv_3x1x1_bn(planes, planes, groups=self.x)
        self.sx_attention = SpatialGroupAttention(planes=planes, div=4, groups=self.x)
        self.tx_attention = TemporalGroupAttention(planes=planes, div=4, groups=self.x)
        self.cx_attention = ChannelGroupAttention(planes=planes, div=4, groups=self.x)
        self.fusion = conv_1x1x1_bn(planes, planes)
        self.conv2_y = conv_1x3x3_bn(planes, planes, groups=self.y)
        self.conv2_z = conv_3x1x1_bn(planes, planes, groups=self.y)
        self.sy_attention = SpatialGroupAttention(planes=planes, div=4, groups=self.y)
        self.ty_attention = TemporalGroupAttention(planes=planes, div=4, groups=self.y)
        self.cy_attention = ChannelGroupAttention(planes=planes, div=4, groups=self.y)

    def forward(self, x):
        nt, c, h, w = x.size()
        n = nt // self.t

        out = x.view(n, self.t, self.x, self.y, h, w)
        out = out.permute(0, 2, 3, 1, 4, 5).contiguous()
        out = out.view(n, c, self.t, h, w)
        out = self.cx_attention(self.sx_attention(self.conv1_x(out)) + self.tx_attention(self.conv1_z(out)))

        out = self.fusion(out)

        out = out.view(n, self.x, self.y, self.t, h, w)
        out = out.permute(0, 2, 1, 3, 4, 5).contiguous()
        out = out.view(n, c, self.t, h, w)
        out = self.cy_attention(self.sy_attention(self.conv2_y(out)) + self.ty_attention(self.conv2_z(out)))

        out = out.view(n, self.y, self.x, self.t, h, w)
        out = out.permute(0, 3, 2, 1, 4, 5).contiguous()
        out = out.view(nt, c, h, w)

        return out


class SpatialGroupAttention(nn.Module):
    def __init__(self, planes, div=4, groups=8):
        super(SpatialGroupAttention, self).__init__()
        self.c = planes // div
        self.fc1 = conv_1x3x3_bn(planes, self.c, groups=groups)
        self.fc2 = conv_1x3x3(self.c, planes, groups=groups)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        n, c, t, h, w = x.size()

        out = F.avg_pool3d(x, kernel_size=[t, 1, 1])

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return x * out


class TemporalGroupAttention(nn.Module):
    def __init__(self, planes, div=4, groups=8):
        super(TemporalGroupAttention, self).__init__()
        self.c = planes // div
        self.fc1 = conv_3x1x1_bn(planes, self.c, groups=groups)
        self.fc2 = conv_3x1x1(self.c, planes, groups=groups)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        n, c, t, h, w = x.size()

        out = F.avg_pool3d(x, kernel_size=[1, h, w])

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return x * out


class ChannelGroupAttention(nn.Module):
    def __init__(self, planes, div=4, groups=8):
        super(ChannelGroupAttention, self).__init__()
        self.c = planes // div
        self.fc1 = conv_1x1x1_bn(planes, self.c, groups=groups)
        self.fc2 = conv_1x1x1(self.c, planes, groups=groups)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        n, c, t, h, w = x.size()

        out = F.avg_pool3d(x, kernel_size=[1, h, w])

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return x * out