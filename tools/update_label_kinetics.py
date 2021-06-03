# -*- coding: utf-8 -*-
#  @Author: KunchangLi
#  @Date: 2020-02-08 20:17:05
#  @LastEditor: KunchangLi
#  @LastEditTime: 2020-02-09 22:57:39

import os
from p_tqdm import p_imap

ROOT_DATASET = '/data1/ckli/kinetics_rgb_img_256_340'
val_label_path = './data/kinetics/kinetics_val_list.txt'
train_label_path = './data/kinetics/kinetics_train_list.txt'

def update_line(line):
    path, _, index = line.strip('\n').split(' ')
    total_path = os.path.join(ROOT_DATASET, path)
    length = len(os.listdir(total_path))
    new_line = '%s %d %s' % (path, length, index)
    return new_line

def update_label_file(label_path):
    print("Update ", label_path.split('\n')[-1])

    with open(label_path) as f:
        lines = f.readlines()
    
    output = []

    iterator = p_imap(update_line, lines, position=0, leave=True, ncols=100, dynamic_ncols=False)
    for result in iterator:
        output.append(result)
    
    print("Start writing")
    with open(label_path,'w') as f:
        f.write('\n'.join(output))
    print("End writing")

if __name__ == '__main__':
    # update_label_file(val_label_path)
    update_label_file(train_label_path)
    