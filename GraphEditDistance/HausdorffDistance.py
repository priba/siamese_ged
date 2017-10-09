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
from torch.autograd.variable import Variable

# Own modules

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat" 


class Hd(nn.Module):

    # Constructor
    def __init__(self, args={}):
        super(Hd, self).__init__()
        self.args = args
    
    def forward(self, v1, sz1, v2, sz2):
        d = torch.zeros(v1.size(0), v2.size(0))
        if v1.is_cuda:
            d = d.cuda()
        d = Variable(d)

        for i in range(v1.size(0)):
            x = v1[i, :sz1[i]]
            for j in range(v2.size(0)):
                y = v2[j, :sz2[j]]

                xx = torch.stack([x]*y.size(0))
                yy = torch.stack([y]*x.size(0)).transpose(0,1)

                dxy = torch.sqrt(torch.sum((xx-yy)**2,2))

                m1, _ = dxy.min(dim=1)
                m2, _ = dxy.min(dim=0)
                d[i,j] = torch.max(m1.max(), m2.max())
        return d
