#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    HaussdorfDistance.py: Computes the Hausdorff between the graph nodes.

    * Bibliography: Fischer et al. (2015) "Approximation of graph edit distance based on Hausdorff matching."

    Usage:
"""

from __future__ import print_function

import torch
import torch.nn as nn

# Own modules

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat" 


class Hd(nn.Module):

    # Constructor
    def __init__(self, args={}):
        super(Hd, self).__init__()
        self.args = args
    
    def forward(self, v1, v2):
        d = 0
        return d
