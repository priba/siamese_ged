#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Ggnn.py: Update function following the Gated Graph Neural Network (GGNN) framework.

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

    # Readout function
    def forward(self, h_v, h_w, e_vw, args=None):
        return output

    # Get the name of the used message function
    def get_definition(self):
        return 'GGNN (Li et al. 2016)'

    # Get the message function arguments
    def get_args(self):
        return self.args

    # Get Output size
    def get_out_size(self, size_h, size_e, args=None):
        return self.m_size(size_h, size_e, args)

