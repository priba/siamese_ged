#-*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np

"""
Useful data tools.
"""

__author__ = 'Pau Riba'
__email__ = 'priba@cvc.uab.cat'


def normalize(data, dim=0):
    mean_data = data.mean(dim)
    std_data = data.std(dim)
    return (data-mean_data.expand_as(data))/std_data.expand_as(data)


def distance(p1, p2):
    "Computes the L2 distance between p1 and p2"
    return np.linalg.norm(p1 - p2)


def angle_between(p1, p2):
    "Computes the angle between p1 and p2, radiants between -pi and pi"
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return ang1 - ang2

