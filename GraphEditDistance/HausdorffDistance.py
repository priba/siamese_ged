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
import numpy as np
# Own modules

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat" 


class Hd(nn.Module):

    # Constructor
    def __init__(self, args={}):
        super(Hd, self).__init__()
        self.args = args
    
    def forward(self, v1, am1, sz1, v2, am2, sz2):

        byy = v2.unsqueeze(1).expand((v2.size(0), v1.size(1), v2.size(1), v2.size(2))).transpose(1,2)
        bxx = v1.unsqueeze(1).expand_as(byy)

        bdxy = torch.sqrt(torch.sum((bxx-byy)**2, 3))

        # Create a mask for nodes
        node_mask2 = torch.arange(0, bdxy.size(1)).unsqueeze(0).unsqueeze(-1).expand(bdxy.size(0),
                                                                                                  bdxy.size(1),
                                                                                                  bdxy.size(2)).long()
        node_mask1 = torch.arange(0, bdxy.size(2)).unsqueeze(0).unsqueeze(0).expand(bdxy.size(0),
                                                                                                 bdxy.size(1),
                                                                                                 bdxy.size(2)).long()

        if v1.is_cuda:
            node_mask1 = node_mask1.cuda()
            node_mask2 = node_mask2.cuda()

        node_mask1 = (node_mask1 >= sz1.unsqueeze(-1).unsqueeze(-1).expand_as(node_mask1))
        node_mask2 = (node_mask2 >= sz2.unsqueeze(-1).unsqueeze(-1).expand_as(node_mask2))

        node_mask = Variable(node_mask1 | node_mask2, requires_grad=False)

        maximum = bdxy.max()

        #bdxy.masked_fill_(node_mask, float(maximum.data.cpu().numpy()[0]))
        bdxy.masked_fill_(node_mask, float(np.inf))

        bm1, _ = bdxy.min(dim=2)
        bm2, _ = bdxy.min(dim=1)

        bm1.masked_fill_(node_mask.prod(dim=2), 0)
        bm2.masked_fill_(node_mask.prod(dim=1), 0)

        d = torch.max(bm1.max(dim=1)[0], bm2.max(dim=1)[0])

        return d

