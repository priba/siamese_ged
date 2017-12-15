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

    # Message from h_v to h_w through e_vw
    # M_t(h^t_v, h^t_w, e_vw) = A_e_vw h^t_w
    def forward(self, h_v, h_w, e_vw):

        m_new = Variable(torch.zeros(h_w.size(0), h_w.size(1), self.args['out']).type_as(h_w.data))

        for i in range(len(self.e_label)):
            edge_labels = torch.nonzero(e_vw.squeeze(2) == self.e_label[i])
            if edge_labels.size():
                parameter_mat = self.edge_matix[i].unsqueeze(0).expand(edge_labels.size(0), self.edge_matix.size(1), self.edge_matix.size(2))

                m_new[edge_labels[:,0],edge_labels[:,1],:] = parameter_mat.bmm(h_w[edge_labels[:,0],edge_labels[:,1],:].unsqueeze(2))

        # e_aux = e_vw.clone()
        # for i in range(len(self.e_label)):
        #     e_aux.masked_fill_(e_vw==self.e_label[i], i)
        # e_aux = e_aux.squeeze().long()
        # edge_output = torch.index_select(self.edge_matix, 0, e_aux)
        #
        # h_w_rows = h_w.unsqueeze(1)
        # h_w_rows = h_w_rows.expand(h_w.size(0), h_v.size(1), h_w.size(1))
        # h_w_rows = h_w_rows.contiguous()
        #
        # h_w_rows = h_w_rows.view(-1, self.in_size)
        #
        # h_multiply = torch.bmm(edge_output, h_w_rows.unsqueeze(2))
        #
        # m_new = h_multiply.squeeze(-1)

        return m_new

    def out_ggnn(self, size_h, size_e, args):
        return self.args['out']

    # Get the name of the used message function
    def get_definition(self):
        return 'GGNN (Li et al. 2016)' 

    # Get the message function arguments
    def get_args(self):
        return self.args
