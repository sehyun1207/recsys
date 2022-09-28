import math
import warnings
import numbers
from typing import List, Tuple, Optional, overload
import os
import time
import random
import joblib
from os.path import abspath, dirname, join as pjoin

import fire
import torch
import  torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch import _VF
import sys
sys.path.append(abspath(dirname(dirname(__file__))))

from base import Model
from loader import SessionDataset, SessionDataLoader
from torch import nn
from torch.nn import Module
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence

def flatten(t):
    return [item for sublist in t for item in sublist]


class RNN(Model):
    def __init__(self, idmap, num_layer=1024, dropout=0.1, dim=64, **kwargss):
        super(RNN, self).__init__()
        self.idmap = idmap
        self.mode = "RNN_TANH"
        self.hidden_size = 64
        self.num_layer = num_layer
        self.n_items = len(self.idmap)
        self.batch_first = False
        self.item_bias = nn.Parameter(torch.zeros(self.n_items).float(),requires_grad=True)
        self.item_emb = nn.Embedding(self.n_items + 1, dim, padding_idx = self.n_items)
        self.drop = nn.Dropout(dropout)
        self.bidirectional = True
        self.init_weights()

        gate_size = self.hidden_size


        self._flat_weights_names = []
        self._all_weights = []

    def forward(self, input, scalar_feats, norm_user=False):
        orig_input = input
        hx=None
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            is_batched = len(input) == 3
            batch_dim = 0 if self.batch_first else 1
            '''if not is_batched:
                if hx is not None:
                    hx = hx.unsqueeze(1)'''
            max_batch_size = len(input)
            sorted_indices = None
            unsorted_indices = None
        num_directions = 2 if self.bidirectional else 1
        hx = torch.zeros(self.num_layer * num_directions, max_batch_size, self.hidden_size)

        
        if batch_sizes is None:
            result = _VF.rnn_tanh(input, hx, self._flat_weights, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.rnn_tanh(input, batch_sizes, hx, self._flat_weights, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional)
        
        output = result[0]
        hidden = result[1]

        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed
        '''if not is_batched:
            output = output.squeeze(batch_dim)
            hidden = hidden.squeeze(1)'''
        
        if norm_user:
            output = F.normalize(output)


        return output


    def init_weights(self):
        for name, weight in self.named_parameters():
            if len(weight.shape) > 1 and 'weight' in name:
                nn.init.xavier_uniform_(weight)
            elif 'bias' in name:
                weight.data.normal_(0.0, 0.001)
                torch.clamp(weight.data, min=-0.001, max=0.001)
