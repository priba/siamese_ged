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
from torch.autograd.variable import Variable

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

        edge_mask = ((e_vw != 0).float().sum(-1) > 0).float()

        m_new = Variable(torch.zeros(h_w.size(0), h_w.size(1), self.args['out']).type_as(h_w.data))

        edge_index = torch.nonzero(edge_mask)
        if edge_index.size():
            parameter_mat = self.edge_matrix(e_vw[edge_index[:, 0], edge_index[:, 1]])
            parameter_mat = parameter_mat.view(-1, self.out_size, self.in_size)

            m_new[edge_index[:, 0], edge_index[:, 1], :] = parameter_mat.bmm(
                h_w[edge_index[:, 0], edge_index[:, 1], :].unsqueeze(2))

        return m_new

    def out_ggnn(self, size_h, size_e, args):
        return self.args['out']

    # Get the name of the used message function
    def get_definition(self):
        return 'EdgeNetwork (Gilmer et al. 2017)' 

    # Get the message function arguments
    def get_args(self):
        return self.args
