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
from torch.autograd.variable import Variable

# Own modules

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat" 


class Ggnn(nn.Module):

    # Constructor
    def __init__(self, args={}):
        super(Ggnn, self).__init__()
        self.args = args
        self.e_label = args['e_label']
        self.in_size = args['in']
        self.out_size = args['out']

        self.edge_matix = nn.Parameter(torch.randn(len(self.e_label), self.out_size, self.in_size))

        self.drop = nn.Dropout(p=0.2)

    # Message from h_v to h_w through e_vw
    # M_t(h^t_v, h^t_w, e_vw) = A_e_vw h^t_w
    def forward(self, h_v, h_w, e_vw):

        m_new = Variable(torch.zeros(h_w.size(0), h_w.size(1), self.args['out']).type_as(h_w.data))

        for i in range(len(self.e_label)):
            edge_labels = torch.nonzero(e_vw.squeeze(2) == self.e_label[i])
            if edge_labels.size():
                parameter_mat = self.edge_matix[i].unsqueeze(0).expand(edge_labels.size(0), self.edge_matix.size(1), self.edge_matix.size(2))

                parameter_mat = self.drop(parameter_mat)

                m_new[edge_labels[:,0],edge_labels[:,1],:] = parameter_mat.bmm(h_w[edge_labels[:,0],edge_labels[:,1],:].unsqueeze(2))

        return m_new

    def out_ggnn(self, size_h, size_e, args):
        return self.args['out']

    # Get the name of the used message function
    def get_definition(self):
        return 'GGNN (Li et al. 2016)' 

    # Get the message function arguments
    def get_args(self):
        return self.args
