# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Load the corresponding dataset.
"""

import datasets
import numpy as np
import glob

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


def load_data(dataset, data_path):
    if dataset == 'letters':
        return load_letters(data_path)
    raise NameError(dataset + ' not implemented!')


def load_washington(data_path):
    # Get data for train, validation and test
    data_train = datasets.Letters(data_path, 'train.cxl')
    data_valid = datasets.Letters(data_path, 'valid.cxl')
    data_test = datasets.Letters(data_path, 'test.cxl')

    return data_train, data_valid, data_test
