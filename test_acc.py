# -*- coding: utf-8 -*-
#  @Author: KunchangLi
#  @Date: 2020-02-11 11:26:51
#  @LastEditor: KunchangLi
#  @LastEditTime: 2020-03-19 08:02:47
#  @Oringin: https://github.com/mit-han-lab/temporal-shift-module/blob/master/main.py

from collections import OrderedDict
import math
import os
import time
import shutil

import numpy as np
from tensorboardX import SummaryWriter
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
import torchvision
from tqdm import tqdm

from ops import (
    TSNDataSet, TSN, return_dataset,
    AverageMeter, accuracy, make_temporal_pool,
    GroupNormalize, GroupScale, GroupCenterCrop, 
    IdentityTransform, Stack, ToTorchFormatTensor,
    GroupFullResSample, GroupOverSample
)
from opts import parser


def main():
    best_prec1 = 0

    num_class, args.train_list, args.val_list, args.root_path, prefix = return_dataset(args.dataset, args.modality)

    model = TSN(num_class, args.num_segments, args.modality, args.model,
                backbone=args.arch, consensus_type=args.consensus_type,
                dropout=args.dropout, partial_bn=not args.no_partialbn,
                pretrain=args.pretrain, pretrain_from=args.pretrain_from,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place, 
                aggregation=args.aggregation, diff_div=args.diff_div, 
                num_total=args.num_total, num_pretrain=args.num_pretrain, group_x=args.group_x,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local, full_res=args.full_res)
        
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_size = model.input_size
    input_mean = model.input_mean
    input_std = model.input_std
    special_ids = model.special_ids
    # train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    
    if not args.resume:
        print('Data Parallel')
        model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
        print('ok')
                                
    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))

            pretrained_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                name = k[7:]  # remove 'module.'
                # name = name.replace('.net', '')
                if 'total' not in name:
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            print('Data Parallel')
            model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
            print('ok')

            if 'epoch' in checkpoint.keys():
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                print(("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.evaluate, checkpoint['epoch'])))
                print(("=> best top1 '{}'".format(best_prec1)))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 3:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(input_size, scale_size, flip=False)
        ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, scale_size, flip=False)
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, scale_size)
        ])
    else:
        raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))

    num_crop = args.test_crops
    if args.dense_sample:
        # num_crop *= 10
        num_crop *= 4
    if args.twice_sample:
        num_crop *= 2

    print(input_size, scale_size, num_crop, args.test_crops)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   test_mode=True,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(),
                       ToTorchFormatTensor(),
                       normalize,
                   ]), dense_sample=args.dense_sample, twice_sample=args.twice_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    if args.evaluate:
        validate(val_loader, model, criterion, 0, num_crop=num_crop)
        return


def validate(val_loader, model, criterion, epoch, num_crop, log=None, tf_writer=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    output = '--------------- Test ---------------'
    print(output)
    if log is not None:
        log.write(output + '\n')
    pbar = tqdm(val_loader, ncols=120)
    with torch.no_grad():
        for i, (input, target) in enumerate(pbar):
            target = target.cuda()

            # compute output
            output = model(input, num_crop)
            output = output.cuda()

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            if i % args.print_freq == 0:
                output = ('Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              loss=losses, top1=top1, top5=top5))
                pbar.set_description(output)
                if log is not None and i % args.log_freq == 0:
                    log.write(output + '\n')
                    log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return top1.avg


if __name__ == '__main__':
    args = parser.parse_args()
    main()
