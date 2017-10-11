#-*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Useful data tools.
"""

__author__ = 'Pau Riba'
__email__ = 'priba@cvc.uab.cat'


def normalize(data, dim=0):
    mean_data = data.mean(dim)
    std_data = data.std(dim)
    return (data-mean_data.expand_as(data))/std_data.expand_as(data)

