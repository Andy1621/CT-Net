# -*- coding: utf-8 -*-
#  @Author: KunchangLi
#  @Date: 2020-02-09 16:35:49
#  @LastEditor: KunchangLi
#  @LastEditTime: 2020-05-03 10:42:48
#  @Oringin: https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/dataset_config.py

import os

__all__ = ['return_dataset']


ROOT_DATASET = '/data1/ckli'
ROOT_LABEL = './data'


def return_ucf101_1(modality):
    n_class = 101
    if modality == 'RGB':
        root_data = 'ucf101/data'
        filename_imglist_train = 'ucf101/ucf101_train_split1_list.txt'
        filename_imglist_val = 'ucf101/ucf101_val_split1_list.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return n_class, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_ucf101_2(modality):
    n_class = 101
    if modality == 'RGB':
        root_data = 'ucf101/data'
        filename_imglist_train = 'ucf101/ucf101_train_split2_list.txt'
        filename_imglist_val = 'ucf101/ucf101_val_split2_list.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return n_class, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_ucf101_3(modality):
    n_class = 101
    if modality == 'RGB':
        root_data = 'ucf101/data'
        filename_imglist_train = 'ucf101/ucf101_train_split3_list.txt'
        filename_imglist_val = 'ucf101/ucf101_val_split3_list.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return n_class, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51_1(modality):
    n_class = 51
    if modality == 'RGB':
        root_data = 'hmdb51'
        filename_imglist_train = 'hmdb51/train_rgb_split1.txt'
        filename_imglist_val = 'hmdb51/val_rgb_split1.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return n_class, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51_2(modality):
    n_class = 51
    if modality == 'RGB':
        root_data = 'hmdb51'
        filename_imglist_train = 'hmdb51/train_rgb_split2.txt'
        filename_imglist_val = 'hmdb51/val_rgb_split2.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return n_class, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51_3(modality):
    n_class = 51
    if modality == 'RGB':
        root_data = 'hmdb51'
        filename_imglist_train = 'hmdb51/train_rgb_split3.txt'
        filename_imglist_val = 'hmdb51/val_rgb_split3.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return n_class, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality):
    n_class = 174
    if modality == 'RGB':
        root_data = 'sthv1'
        filename_imglist_train = 'something/train_videofolder.txt'
        filename_imglist_val = 'something/val_videofolder.txt'
        # filename_imglist_val = 'something/val_videofolder2.txt'
        prefix = '{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return n_class, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    n_class = 174
    if modality == 'RGB':
        root_data = 'sthv2/20bn-something-something-v2-frames'
        filename_imglist_train = 'somethingv2/train.txt'
        filename_imglist_val = 'somethingv2/val.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return n_class, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    n_class = 174
    if modality == 'RGB':
        root_data = 'sthv2/20bn-something-something-v2-frames'
        filename_imglist_train = 'somethingv2/train.txt'
        filename_imglist_val = 'somethingv2/val.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return n_class, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    n_class = 400
    if modality == 'RGB':
        root_data = 'kinetics'
        filename_imglist_train = 'kinetics/train_list.txt'
        filename_imglist_val = 'kinetics/val_list.txt'
        prefix = 'image_{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return n_class, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'ucf101_1': return_ucf101_1, 
                   'ucf101_2': return_ucf101_2,
                   'ucf101_3': return_ucf101_3, 
                   'hmdb51_1': return_hmdb51_1, 
                   'hmdb51_2': return_hmdb51_2,
                   'hmdb51_3': return_hmdb51_3,
                   'something': return_something,
                   'somethingv2': return_somethingv2,
                   'kinetics': return_kinetics }
    if dataset in dict_single:
        n_class, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    root_data = os.path.join(ROOT_DATASET, root_data)
    file_imglist_train = os.path.join(ROOT_LABEL, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_LABEL, file_imglist_val)
    print('{}: {} classes'.format(dataset, n_class))

    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
