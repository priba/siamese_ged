#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    HaussdorfEditdistance.py: Computes an aproximated Graph Edit Distance.

    * Bibliography: Fischer et al. (2015) "Approximation of graph edit distance based on Hausdorff matching."

    Usage:
"""

from __future__ import print_function

import torch
import torch.nn as nn

# Own modules

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat" 


class Hed(nn.Module):

    # Constructor
    def __init__(self, args={}):
        super(Hed, self).__init__()
        self.args = args
    
        self.insertion = None
        self.deletion = None
        self.substitution = None

    def forward(self, v1, am1, v2, am2):
        d = 0
        return d
