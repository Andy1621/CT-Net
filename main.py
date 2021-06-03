# -*- coding: utf-8 -*-
#  @Author: KunchangLi
#  @Date: 2020-02-11 11:26:51
#  @LastEditor: KunchangLi
#  @LastEditTime: 2020-05-04 11:59:22
#  @Oringin: https://github.com/mit-han-lab/temporal-shift-module/blob/master/main.py

from collections import OrderedDict
import math
import os
import time
import shutil

import numpy as np
from tensorboardX import SummaryWriter
from thop import profile
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
    IdentityTransform, Stack, ToTorchFormatTensor
)
from opts import parser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_store_name():
    full_arch_name = args.arch
    if args.model == 'TSM':
        assert args.shift == True, 'TSM should set parameter "shift"'
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    elif args.model in ['VRM', 'STM']:
        full_arch_name += '_reduce{}_{}'.format(args.diff_div, args.aggregation)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        [args.model, args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if not args.pretrain:
        args.store_name += '_npre'
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if not args.no_partialbn:
        args.store_name += '_pbn'
    if args.suffix:
        args.store_name += '_{}'.format(args.suffix)
    else:
        if args.resume:
            total_name = args.resume.split('/')[-2]
            suffix = total_name.split('_')[-1]
        else:
            suffix = round(time.time())
        args.store_name += '_{}'.format(suffix)
    print('storing name: ' + args.store_name)
    

def load_process(state_dict):
    # remove '.module'
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict


def get_optim_policies(model, special_ids=None, unfrozen=False):
    if args.model == 'VRM':
        if unfrozen:
            print('Unfrozen...')
            extra_id = list()
            for p in model.parameters():
                if p.requires_grad:
                    extra_id.append(id(p))
                else:
                    if id(p) not in special_ids:
                        p.requires_grad = True
            policies = model.module.get_optim_policies(extra_id=extra_id)
        else:
            special_group = filter(lambda p: p.requires_grad, model.parameters())
            policies = [{'params': special_group, 'lr': args.lr}]
            return policies
    else:
        policies = model.module.get_optim_policies()
    for group in policies:
        out = 'group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])
        if 'lr' in group:
            out += ', lr: {}'.format(group['lr'])
        print(out)
    return policies


def count_model(model, crop_size):
    input = torch.randn(1, 3 * args.num_segments, crop_size, crop_size)
    flops, params = profile(model, inputs=(input, ))
    print("flops: ", flops, ", params: ", params)


def main():
    seed = 42
    setup_seed(seed)

    best_prec1 = 0

    num_class, args.train_list, args.val_list, args.root_path, prefix = return_dataset(args.dataset, args.modality)
    
    get_store_name()
    check_rootfolders()

    # From https://github.com/swathikirans/GSM/blob/43e8ebad5cf1bf2aaca1674a753a89fcba416321/main.py#L57
    # if 'something' in args.dataset:
    # if args.dataset == 'something':
    #     # label transformation for left/right categories
    #     target_transform = {86:87,87:86,93:94,94:93,166:167,167:166}
    #     print('Target transformation is enabled....')
    # else:
    target_transform = None

    model = TSN(num_class, args.num_segments, args.modality, args.model,
                backbone=args.arch, consensus_type=args.consensus_type,
                dropout=args.dropout, partial_bn=not args.no_partialbn,
                pretrain=args.pretrain, pretrain_from=args.pretrain_from,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place, 
                aggregation=args.aggregation, diff_div=args.diff_div, 
                num_total=args.num_total, num_pretrain=args.num_pretrain, group_x=args.group_x,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local, full_res=args.full_res, target_transform=target_transform)

    count_model(model, model.crop_size)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    special_ids = model.special_ids
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)
    
    if not args.resume:
        print('Data Parallel')
        model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
        print('ok')

        optimizer = torch.optim.SGD(get_optim_policies(model, special_ids=special_ids, unfrozen=True), 
                                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
                                
    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))

            pretrained_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                if '.total' not in k:
                    name = k[7:]  # remove 'module.'
                    # name = name.replace('.net', '')
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            print('Data Parallel')
            model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
            print('ok')

            if 'epoch' in checkpoint.keys():
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                
                optimizer = torch.optim.SGD(get_optim_policies(model, special_ids=special_ids, unfrozen=True), 
                                            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

                optimizer.load_state_dict(checkpoint['optimizer'])
                print(("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.evaluate, checkpoint['epoch'])))
                print(("=> best top1 '{}'".format(best_prec1)))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)

        new_state_dict = OrderedDict()
        for k, v in sd.items():
            if '.total' not in k:
                new_state_dict[k] = v
        sd = new_state_dict
        
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k and '.total' not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

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

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   seed = seed,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(),
                       ToTorchFormatTensor(),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   seed =seed,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(),
                       ToTorchFormatTensor(),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.txt'), 'a')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, epoch, log_training, tf_writer)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

            output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    output = '--------------- Epoch: [{epoch}], lr: {lr:.5f} ---------------'.format( epoch=epoch, lr=optimizer.param_groups[-1]['lr'])
    print(output)
    log.write(output + '\n')
    pbar = tqdm(train_loader, ncols=120)
    for i, (input, target) in enumerate(pbar):
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        if i % args.print_freq == 0:
            output = ('Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          loss=losses, top1=top1, top5=top5))
            pbar.set_description(output)
            if i % args.log_freq == 0:
                log.write(output + '\n')
                log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    throughput = AverageMeter()

    # switch to evaluate mode
    model.eval()

    output = '--------------- Test ---------------'
    print(output)
    if log is not None:
        log.write(output + '\n')
    pbar = tqdm(val_loader, ncols=130)
    with torch.no_grad():
        for i, (input, target) in enumerate(pbar):
            target = target.cuda()

            # compute output

            torch.cuda.synchronize()
            start = time.time()

            output = model(input)

            torch.cuda.synchronize()
            end = time.time()

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            videos_times_second = input.size(0) / (end - start)
            throughput.update(videos_times_second)

            if i % args.print_freq == 0:
                output = ('Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                          'throughput {throughput.avg:.3f}'.format(
                              loss=losses, top1=top1, top5=top5, throughput=throughput))
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


def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
    # save special epoch
    epoch = state['epoch']
    # if epoch == 10 or epoch == 35 or epoch == 50 or epoch == 55:
    if epoch == 10:
        shutil.copyfile(filename, filename.replace('pth.tar', 
                    'backup_e{}.pth.tar'.format(epoch - 1)))
    if epoch > args.epochs - 10:
        shutil.copyfile(filename, filename.replace('pth.tar', 
                    'backup_e{}.pth.tar'.format(epoch - 1)))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    if epoch < args.warmup:
        lr = lr * (epoch + 1) / args.warmup
    else:
        if lr_type == 'step':
            lr = lr * 0.1 ** (sum(epoch >= np.array(lr_steps)))
        elif lr_type == 'cos':
            if epoch < args.tune_epoch:
                cos_decay = 0.5 * (1 + math.cos(math.pi * (epoch - args.warmup) / (args.tune_epoch - args.warmup)))
                lr = args.tune_lr + cos_decay * (lr - args.tune_lr)
            else:
                # if +1, it will end up with 0
                cos_decay = 0.5 * (1 + math.cos(math.pi * (epoch - args.tune_epoch) / (args.epochs - args.tune_epoch)))
                lr = args.tune_lr * cos_decay
        else:
            raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = param_group['weight_decay'] * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    args = parser.parse_args()
    main()
