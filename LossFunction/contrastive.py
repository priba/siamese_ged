# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Contrastive loss function.

Based on: https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py
"""

import torch
import torch.nn

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, y):

        mdist = self.margin - dist
        mdist = torch.clamp(mdist, min=0.0)
        loss = y * dist.pow(2) + (1 - y) * torch.pow(mdist, 2)
        loss = (loss.sum() / 2.0) / y.size(0)
        return loss

