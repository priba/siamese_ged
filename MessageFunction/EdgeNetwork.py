#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    EdgeNetwork.py: Propagates a message following the Gated Graph Neural Network (GGNN) modification proposed by Gilmer.

    * Bibliography: Gilmer et al. (2017), Neural Message Passing for Quantum Chemistry

    Usage:
"""

from __future__ import print_function

import torch
import torch.nn as nn
# Own modules

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat" 


class EdgeNetwork(nn.Module):

    # Constructor
    def __init__(self, args={}):
        super(EdgeNetwork, self).__init__()
        self.args = args
        self.e_size = args['e_size']
        self.in_size = args['in']
        self.out_size = args['out']
        if self.out_size*self.in_size > 64:
            self.edge_matrix = nn.Sequential(nn.Linear(self.e_size, 128), nn.ReLU(),
                    nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 2048), nn.ReLU(),
                    nn.Linear(2048, self.out_size*self.in_size))
        else:
            self.edge_matrix = nn.Sequential(nn.Linear(self.e_size, self.out_size*self.in_size), nn.Tanh())


    # Message from h_v to h_w through e_vw
    # M_t(h^t_v, h^t_w, e_vw) = A_e_vw h^t_w
    def forward(self, h_v, h_w, e_vw):
        edge_output = self.edge_matrix(e_vw)
        edge_output = edge_output.view(-1, self.out_size, self.in_size)

        h_w_rows = h_w[..., None].expand(h_w.size(0), h_v.size(1), h_w.size(1)).contiguous()

        h_w_rows = h_w_rows.view(-1, self.in_size)

        h_multiply = torch.bmm(edge_output, h_w_rows.unsqueeze(2))

        m_new = h_multiply.squeeze(-1)

        return m_new

    def out_ggnn(self, size_h, size_e, args):
        return self.args['out']

    # Get the name of the used message function
    def get_definition(self):
        return 'EdgeNetwork (Gilmer et al. 2017)' 

    # Get the message function arguments
    def get_args(self):
        return self.args
