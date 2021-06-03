# -*- coding: utf-8 -*-
#  @Author: KunchangLi
#  @Date: 2020-02-09 16:03:29
#  @LastEditor: KunchangLi
#  @LastEditTime: 2020-03-18 17:38:56
#  @Oringin: https://github.com/mit-han-lab/temporal-shift-module/blob/master/opts.py

import argparse

parser = argparse.ArgumentParser(description='PyTorch implementation of Temporal Segment Networks')
parser.add_argument('dataset', type=str, default=None, choices=['kinetics', 'something', 'somethingv2', 'ucf101_1', 'ucf101_2', 'ucf101_3', 'hmdb51_1', 'hmdb51_2', 'hmdb51_3'], help='name of dateset(kinetics/something/somethingv2/ucf101_1/ucf101_2/ucf101_3/hmdb51_1/hmdb51_2/hmdb51_3) (default: None)')
parser.add_argument('modality', type=str, default=None, choices=['RGB', 'Flow', 'RGBDiff'],
                    help='modalit fo dataset(RGB/Flow/RGBDiff) (default: None)')
parser.add_argument('--train-list', type=str, default=None,
                    help='list of train data (default: None)')
parser.add_argument('--val-list', type=str, default=None,
                    help='list of val data (default: None)')
parser.add_argument('--root-path', type=str, default=None,
                    help='root path to data, (default: None)')
parser.add_argument('--root-log',type=str, default='log',
                    help='root path to log (default: log)')
parser.add_argument('--root-model', type=str, default='checkpoint',
                    help='root path to checkpoint (default: checkpoint)')
parser.add_argument('--store-name', type=str, default=None,
                    help='name of experiment (default: None)')
parser.add_argument('--suffix', type=str, default=None,
                    help='suffix of the name of experiment (default: None)')
parser.add_argument('--pkl', type=str, default=None,
                    help='name of pkl (default: None)')

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default='resnet50', 
                    help='name of backbone (default: resnet50)')
parser.add_argument('--model', type=str, default=None, choices=['TSN', 'TSM', 'CT_Net'], 
                    help='name of model(TSN/TSM/CT_Net) (default: None)')
parser.add_argument('--num-segments', type=int, default=8,
                    help='number of segments) (default: 8)')
parser.add_argument('--consensus-type', type=str, default='avg', 
                    help='type of consensus (default: avg))')
parser.add_argument('--dropout', '--do', type=float, default=0.5,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss-type', type=str, default='nll', choices=['nll'], 
                    help='type of loss function (default: nll)',)
parser.add_argument('--pretrain',  type=str, default='ImageNet', 
                    choices=['ImageNet', 'Something'], help='ImageNet for ResNet, Something for TSM(ImageNet/Something) (default: ImageNet)')
parser.add_argument('--tune-from', type=str, default=None, 
                    help='fine-tune from checkpoint (default: None)')
# for TSM
parser.add_argument('--shift', default=False, action='store_true', 
                    help='use shift for models (default: False)')
parser.add_argument('--shift-div', default=8, type=int, 
                    help='number of div for shift (default: 8)')
parser.add_argument('--shift-place', type=str, default='blockres', 
                    help='place for shift (default: blockres)')
parser.add_argument('--temporal-pool', default=False, action='store_true', 
                    help='add temporal pooling (default: False)')
# for VRM
parser.add_argument('--aggregation', type=str, default='sum', choices=['sum', 'cat', 'mul'], 
                    help='how to arrregation(sum/cat/mul) (default: sum)')
parser.add_argument('--diff-div', default=8, type=int, 
                    help='number of div for diff (default: 8)')
parser.add_argument('--unfrozen-epoch', type=int, default=-1, 
                    help='unfreeze the frozon pretrained layers (default: -1)')
parser.add_argument('--pretrain-from', type=str, default=None,
                    help='path to pretrained model checkpoint (default: None)')
parser.add_argument('--num-total', type=int, default=7,
                    help='number of total block (default: 7)')
parser.add_argument('--num-pretrain', type=int, default=0,
                    help='number of pretrained block (default: 0)')
parser.add_argument('--group_x', type=str, default=None, 
                    help='the number of group_x, such as 4/8/16, (default: None)')

parser.add_argument('--non-local', default=False, action='store_true', 
                    help='add non local block (default: False)')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', type=int, default=120, 
                    metavar='N', help='number of total epochs (default: 120)')
parser.add_argument('-b', '--batch-size', type=int, default=128, 
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate',type=float, default=0.001,  
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--lr-type', type=str, default='step', choices=['step', 'cos'],
                    help='learning rate type (default: step)')
parser.add_argument('--lr-steps', type=int, default=[100], nargs='+',
                    help='epochs to decay learning rate by 10 (default: [100])')
parser.add_argument('--tune-epoch', type=int, default=40,
                    help='epoch to set smaller learning rate (default: 40)')
parser.add_argument('--tune-lr', type=float, default=0.0001,
                    help='the minimum learning rate while fine-tuning (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9, 
                    metavar='M', help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd',type=float, default=5e-4,  
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', type=float, default=None,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no-partialbn', '--npb', default=False, action='store_true',
                    help='disable partialBN (default: False)')
parser.add_argument('--warmup', type=int, default=0, 
                    help='number of warmup epochs (default: 0)')


# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-pf', type=int, default=1,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument('--log-freq', '-lf', type=int, default=20,
                    metavar='N', help='log frequency (default: 20)')
parser.add_argument('--eval-freq', '-ef', type=int, default=1,
                    metavar='N', help='evaluation frequency (default: 1)')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', type=int,  default=8,
                    metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--resume', type=str, default=None,
                    help='path to latest checkpoint (default: None)')
parser.add_argument('-e', '--evaluate', default=False, action='store_true',
                    help='evaluate model (default false)')
parser.add_argument('--start-epoch', type=int, default=0,
                    metavar='N', help='number of start epoch (default 0)')
parser.add_argument('--dense-sample', default=False, action='store_true', 
                    help='enable dense sampling (default False)')
parser.add_argument('--twice-sample', default=False, action='store_true', 
                    help='enable twice sampling (default False)')
parser.add_argument('--test-crops', type=int, default=1)
parser.add_argument('--full-res', default=False, action='store_true',
                    help='enable full resolution (default False)')
parser.add_argument('--gpus', nargs='+', type=int, default=None,
                    help='the No of gpu (default: None)')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')