# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Load the corresponding dataset.
"""

import torch
import datasets
import numpy as np
import glob

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


def load_data(dataset, data_path):
    if dataset == 'letters':
        return load_letters(data_path)
    raise NameError(dataset + ' not implemented!')


def load_letters(data_path):
    # Get data for train, validation and test
    data_train = datasets.Letters(data_path, 'train.cxl')
    data_valid = datasets.Letters(data_path, 'validation.cxl')
    data_test = datasets.Letters(data_path, 'test.cxl')

    return data_train, data_valid, data_test


def collate_fn_multiple_size(batch):
    n_batch = len(batch)
    g_size = torch.LongTensor([x[0].size(0) for x in batch])
    graph_size = torch.LongTensor([[x[0].size(0), x[0].size(1), x[1].size(2)] for x in batch])
    sz, _ = graph_size.max(dim=0)

    n_labels = torch.zeros(n_batch, sz[0], sz[1])
    am = torch.zeros(n_batch, sz[0], sz[0], sz[2])
    targets = torch.LongTensor([x[2] for x in batch])

    for i in range(n_batch):
        # Node Features
        n_labels[i, :g_size[i], :] = batch[i][0]

        # Adjacency matrix
        am[i, :g_size[i], :g_size[i], :] = batch[i][1]

    return n_labels, am, g_size, targets
