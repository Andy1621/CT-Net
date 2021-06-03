# -*- coding: utf-8 -*-
#  @Author: KunchangLi
#  @Date: 2020-02-11 09:42:28
#  @LastEditor: KunchangLi
#  @LastEditTime: 2020-02-11 09:42:29

import torch

__all__ = ['Identity', 'ConsensusModule']


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)
