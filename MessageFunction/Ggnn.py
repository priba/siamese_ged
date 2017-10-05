#!/usr/bin/python                                                                                                                                                                                               
# -*- coding: utf-8 -*-

"""
    Ggnn.py: Propagates a message following the Gated Graph Neural Network (GGNN) framework.

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
        self.e_label = args{'e_label'}
        self.in_size = args{'in'}
        self.out_size = args{'out'}

        self.edge_matix = nn.Parameter(torch.randn(len(self.e_label), self.out_size, self.in_size))

    # Message from h_v to h_w through e_vw
    # M_t(h^t_v, h^t_w, e_vw) = A_e_vw h^t_w
    def forward(self, h_v, h_w, e_vw):
        e_vw = e_vw - 1
        e_vw[e_vw == -1] = 0
        e_vw = torch.squeeze(e_vw.long())
        edge_output = torch.index_select(self.learn_args[0], 0, e_vw)

        h_w_rows = h_w[..., None].expand(h_w.size(0), h_v.size(1), h_w.size(1)).contiguous()

        h_w_rows = h_w_rows.view(-1, self.args['in'])

        h_multiply = torch.bmm(edge_output, torch.unsqueeze(h_w_rows, 2))

        m_new = torch.squeeze(h_multiply)

        return m_new

    def out_ggnn(self, size_h, size_e, args):
        return self.args['out']

    # Get the name of the used message function
    def get_definition(self):
        return 'GGNN (Li et al. 2016)' 

    # Get the message function arguments
    def get_args(self):
        return self.args
