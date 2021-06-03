'''
Author: Kunchang Li
Date: 2021-03-29 19:21:25
LastEditors: Kunchang Li
LastEditTime: 2021-04-01 19:11:40
'''
# -*- coding: utf-8 -*-
#  @Author: KunchangLi
#  @Date: 2020-02-09 22:56:12
#  @LastEditor: KunchangLi
#  @LastEditTime: 2020-05-04 11:51:53

from thop import profile
import torch
from torchsummary import summary

from models import TSN


def print_summary(model):
    summary(model, (24, 224, 224), device='cpu')


def count_model(model, device):
    input = torch.randn(1, 24, 224, 224).to(device)
    flops, params = profile(model, inputs=(input))
    print("flops: ", flops, ", params: ", params)


if __name__ == "__main__":
    net = TSN(174, 8, 'RGB', 'CT_Net', backbone='resnet50', consensus_type='avg', 
        dropout=0.5, partial_bn=False, aggregation='sum', 
        pretrain='ImageNet', num_total=7, diff_div=8, fc_lr5=True, temporal_pool=False, non_local=False)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = net.to(device)
    print(net)

    count_model(model, device)

    # print_summary(net)