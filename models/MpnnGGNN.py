#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable

import MessageFunction
import UpdateFunction
import ReadoutFunction

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


class MpnnGGNN(nn.Module):
    """
        MPNN as proposed by Li et al..

        This class implements the whole Li et al. model following the functions proposed by Gilmer et al. as
        Message, Update and Readout.

        Parameters
        ----------
        in_size : int
            Size of the input features per node.
        e : int list.
            Possible edge labels for the input graph.
        hidden_state_size : int
            Size of the hidden states (the input will be padded with 0's to this size).
        message_size : int
            Message function output vector size.
        n_layers : int
            Number of iterations Message+Update (weight tying).
        target_size : int
            Number of output classes.
        out_type : str (Optional)
            Classification | [Regression (default)]. If classification, LogSoftmax layer is applied to the output vector.
    """

    def __init__(self, in_size, e, hidden_state_size, message_size, n_layers, target_size, out_type='classification'):
        super(MpnnGGNN, self).__init__()

        # Define message
        self.m = MessageFunction.Ggnn(args={'e_label': e, 'in': hidden_state_size, 'out': message_size})

        # Define Update
        self.u = UpdateFunction.Ggnn(args={'in_m': message_size, 'out': hidden_state_size})

        # Define Readout
        self.r = ReadoutFunction.Ggnn(args={'in': in_size, 'hidden': hidden_state_size, 'target': target_size})

        self.type = out_type.lower()

        self.args = {}
        self.args['out'] = hidden_state_size

        self.n_layers = n_layers

    def forward(self, h_in, am, g_size):

        # Padding to some larger dimension d
        h_t = torch.cat([h_in, Variable(
            torch.zeros(h_in.size(0), h_in.size(1), self.args['out'] - h_in.size(2)).type_as(h_in.data))], 2)
        
        # Create a mask for nodes
        node_mask = torch.arange(0,h_in.size(1)).unsqueeze(0).expand(h_in.size(0), h_in.size(1)).long()
        if g_size.is_cuda:
            node_mask = node_mask.cuda()
        node_mask = (node_mask < g_size.unsqueeze(-1).expand_as(node_mask)).float()
        node_mask = Variable(node_mask.unsqueeze(-1))

        # Create a mask for edges
        edge_mask = (am.sum(-1,keepdim=True) > 0).float()

        # Layer
        for t in range(0, self.n_layers):
            e_aux = am.view(-1, am.size(3))

            h_aux = h_t.view(-1, h_t.size(2))

            m = self.m(h_t, h_aux, e_aux)
            m = m.view(h_t.size(0), h_t.size(1), -1, m.size(1))

            # Nodes without edge set message to 0
            m = edge_mask.expand_as(m) * m

            m = m.sum(1)

            h_t = self.u(h_t, m)

            # Delete virtual nodes
            h_t = node_mask.expand_as(h_t) * h_t

        # Readout
        res = self.r([h_t, h_in])

        if self.type == 'classification':
            res = nn.LogSoftmax()(res)
        return res

