#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Ggnn.py: Readout function following the Gated Graph Neural Network (GGNN) framework.

    * Bibliography: Li et al. (2016), Gated Graph Neural Networks (GG-NN)

    Usage:
"""

from __future__ import print_function

import torch
import torch.nn as nn

# Own modules

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


class Ggnn(nn.Module):
    # Constructor
    def __init__(self, args={}):
        super(Ggnn, self).__init__()
        self.args = args
        self.in_size = args['in']
        self.hidden_size = args['hidden']
        self.out_size = args['target']

        self.i = nn.Sequential(nn.Linear(self.in_size+self.hidden_size, 128),
                    nn.ReLU(), nn.Linear(128, self.out_size))
        self.j = nn.Sequential(nn.Linear(self.hidden_size,128), nn.ReLU(), nn.Linear(128,self.out_size))

    # Readout function
    def forward(self, h, args=None):
        h_t = h[0].view(-1, h[0].size(2))
        h_0 = h[1].view(-1, h[1].size(2))
        read = nn.Sigmoid()(self.i(torch.cat([h_t, h_0], 1))*self.j(h_t))

        read = read.view(h[0].size(0), h[0].size(1), -1)

        read = args['node_mask'].expand_as(read) * read
        read = read.sum(1)

        return read

    # Get the name of the used message function
    def get_definition(self):
        return 'GGNN (Li et al. 2016)'

    # Get the message function arguments
    def get_args(self):
        return self.args

    # Get Output size
    def get_out_size(self, size_h, size_e, args=None):
        return self.m_size(size_h, size_e, args)

