# -*- coding: utf-8 -*-
#  @Author: KunchangLi
#  @Date: 2020-02-09 21:09:19
#  @LastEditor: KunchangLi
#  @LastEditTime: 2020-05-05 00:12:49
#  @Oringin: https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/models.py

from collections import OrderedDict

import torch
from torch import nn
from torch.nn.init import normal_, constant_
import torchvision

import arch
from basic_ops import ConsensusModule
from non_local import make_non_local
from temporal_shift import make_temporal_shift

from reshape_block import make_reshape_block  # parallel + x 1x3x3&3x1x1 + y 1x3x3&3x1x1 + 1x1x1 fusion + spatiotemporal group attention plus

from transforms import GroupMultiScaleCrop, GroupRandomHorizontalFlip

__all__ = ['TSN']


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality, model,
                 backbone='resnet50', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, partial_bn=True, print_spec=True, 
                 pretrain='ImageNet', pretrain_from=None,
                 is_shift=False, shift_div=8, shift_place='blockres', 
                 aggregation='sum', diff_div=8, num_total=7, num_pretrain=0,
                 group_x=None, fc_lr5=False, temporal_pool=False, non_local=False, 
                 full_res=False, target_transform=None):
        super(TSN, self).__init__()
        self.modality = modality
        self.model = model
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.consensus_type = consensus_type
        self.pretrain = pretrain
        self.target_transform = target_transform

        # TSM
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        # VRM
        self.aggregation = aggregation
        self.special_ids = list()
        self.diff_div = diff_div
        self.num_total = num_total
        self.num_pretrain = num_pretrain
        self.group_x = group_x
        self.pretrain_from = pretrain_from
        self.pretrain_fc = None

        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local
        self.full_res = full_res
        self.backbone = backbone

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
            """.format(backbone, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(backbone)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax(dim=1)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if self.pretrain == 'Something':
                print("=> loading weight of new_fc")
                fc_dict = self.new_fc.state_dict()
                # print(fc_dict.keys())
                fc_dict.update(self.pretrain_fc)
                self.new_fc.load_state_dict(fc_dict)
                print("=> load ok")
            elif hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, backbone):
        print('=> backbone: {}'.format(backbone))
        print('=> base model: {}'.format(self.model))

        if 'resnet' in backbone or 'inception_v3' in backbone or 'resnext' in backbone or 'xyresnet' in backbone:
            if self.pretrain == 'ImageNet':
                self.base_model = getattr(arch, backbone)(True)
            else:
                raise NotImplementedError

            if self.model == 'TSM':
                print('Adding temporal shift...')
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)
            elif self.model == 'CT_Net':
                print('Freeze the pretrained parameters...')
                for p in self.parameters():
                    p.requires_grad = False
                if 'inception' in backbone:
                    make_reshape_block_for_inception(self.base_model, self.num_total, self.num_segments)
                elif 'resnext' in backbone:
                    make_reshape_block_for_resnext(self.base_model, self.num_total, self.num_segments)
                elif 'wide_resnet' in backbone:
                    make_reshape_block_for_wide_resnet(self.base_model, self.num_total, self.num_segments)
                else:
                    print('Replacing original block to {} new block...'.format(self.num_total))
                    make_reshape_block(self.base_model, self.num_total, self.diff_div, self.num_segments, self.aggregation)
            elif self.model != 'TSN':
                raise NotImplementedError

            if self.non_local:
                print('Adding non-local module...')
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            if self.full_res:
                self.input_size = 256
                if self.backbone == 'inception_v3':
                    self.input_size = 299
                    # self.input_size = 256
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif backbone == 'bninception':
            self.base_model = getattr(arch, backbone)(pretrained=self.pretrain)
            self.input_size = 224
            self.input_mean = self.base_model.mean
            self.input_std = self.base_model.std
            self.base_model.last_layer_name = 'last_linear'
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
            elif self.model == 'VRM':
                make_reshape_block_for_bninception(self.base_model, self.num_total, self.num_segments)

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

            for name, p in self.base_model.named_parameters():
                if 'conv1.1' in name or 'conv3.1' in name or 'special_bn' in name:
                    if p.requires_grad == False:
                        p.requires_grad = True

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self, extra_id=None):
        first_conv_weight = list()
        first_conv_bias = list()
        normal_weight = list()
        normal_bias = list()
        lr5_weight = list()
        lr10_bias = list()
        bn = list()
        custom_ops = list()

        conv_cnt = 0
        bn_cnt = 0

        extra_weight = list()
        extra_bias = list()
        extra_bn = list()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    if id(ps[0]) not in self.special_ids:
                        # not add special convolution whose gradient is not updated
                        if extra_id and id(ps[0]) in extra_id:
                            extra_weight.append(ps[0])
                            if len(ps) == 2:
                                extra_bias.append(ps[1])
                        else:
                            normal_weight.append(ps[0])
                            if len(ps) == 2:
                                normal_bias.append(ps[1])
            elif isinstance(m, nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    ps = list(m.parameters())
                    if extra_id and id(ps[0]) in extra_id:
                        extra_bn.extend(ps)
                    else:
                        bn.extend(ps)
            elif isinstance(m, nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        policies = [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
            'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
            'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
            'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
            'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
            'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
            'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
            'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
            'name': "lr10_bias"},
            # for extra, set lr_mult for different lr
            {'params': extra_weight, 'lr_mult': 1, 'decay_mult': 1,
            'name': "extra_weight"},
            {'params': extra_bias, 'lr_mult': 2, 'decay_mult': 0,
            'name': "extra_bias"},
            {'params': extra_bn, 'lr_mult': 1, 'decay_mult': 0,
            'name': "extra BN scale/shift"},
            # {'params': extra_weight, 'lr_mult': 3, 'decay_mult': 1,
            # 'name': "extra_weight"},
            # {'params': extra_bias, 'lr_mult': 6, 'decay_mult': 0,
            # 'name': "extra_bias"},
            # {'params': extra_bn, 'lr_mult': 3, 'decay_mult': 0,
            # 'name': "extra BN scale/shift"},
            # {'params': extra_weight, 'lr_mult': 5, 'decay_mult': 1,
            # 'name': "extra_weight"},
            # {'params': extra_bias, 'lr_mult': 10, 'decay_mult': 0,
            # 'name': "extra_bias"},
            # {'params': extra_bn, 'lr_mult': 5, 'decay_mult': 0,
            # 'name': "extra BN scale/shift"},
        ]

        return policies

    def forward(self, input, num_crop=1, no_reshape=False):
        if not no_reshape:
            sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

            if self.modality == 'RGBDiff':
                sample_len = 3 * self.new_length
                input = self._get_diff(input)
            base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        else:
            base_out = self.base_model(input)

        features = base_out

        if self.dropout > 0:
            base_out = self.new_fc(base_out)
            
        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.reshape:
            if self.is_shift and self.temporal_pool:
                base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
                features = features.view((-1, self.num_segments // 2) + features.size()[1:])
            else:
                base_out = base_out.view((-1, self.num_segments * num_crop) + base_out.size()[1:])
                features = features.view((-1, self.num_segments  * num_crop) + features.size()[1:])
            output = self.consensus(base_out)
            features = self.consensus(features).view(-1, features.shape[-1], 1, 1)
            return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        # return self.input_size * 256 // 224
        if self.backbone == 'inception_v3':
            return 299
            # return 256
        else:
            return 256

    def get_augmentation(self, flip=True):
        if self.modality == 'RGB':
            if flip:
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]), GroupRandomHorizontalFlip(is_flow=False)])
                # return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]), GroupRandomHorizontalFlip(is_flow=False, target_transform=target_transform)])
            else:
                print('#' * 20, 'NO FLIP!!!')
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]), GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]), GroupRandomHorizontalFlip(is_flow=False)])
