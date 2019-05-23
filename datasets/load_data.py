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


def load_data(dataset, data_path, representation, normalization, siamese=False):
    if dataset == 'letters':
        if siamese:
            return load_letters_siamese(data_path, representation, normalization)
        else:
            return load_letters(data_path, representation, normalization)
    elif dataset == 'histograph':
        if siamese:
            return load_histograph_siamese(data_path, representation, normalization)
        else:
            return load_histograph(data_path, representation, normalization)
    elif dataset == 'histographretrieval':
        if siamese:
            return load_histograph_retrieval_siamese(data_path, representation, normalization)
        else:
            return load_histograph_retrieval(data_path, representation, normalization)
    raise NameError(dataset + ' not implemented!')


def load_letters(data_path, representation='adj', normalization=False):
    # Get data for train, validation and test
    data_train = datasets.Letters(data_path, 'train.cxl', representation, normalization)
    data_valid = datasets.Letters(data_path, 'validation.cxl', representation, normalization)
    data_test = datasets.Letters(data_path, 'test.cxl', representation, normalization)

    return data_train, data_valid, data_test


def load_letters_siamese(data_path, representation='adj', normalization=False):
    # Get data for train, validation and test
    data_train = datasets.LettersSiamese(data_path, 'train.cxl', representation, normalization)
    data_valid = datasets.LettersSiamese(data_path, 'validation.cxl', representation, normalization)
    data_test = datasets.LettersSiamese(data_path, 'test.cxl', representation, normalization)

    return data_train, data_valid, data_test


def load_histograph(data_path, representation='adj', normalization=False):
    # Get data for train, validation and test
    data_train = datasets.HistoGraph(data_path, '../../../../Set/Train.txt', representation, normalization)
    data_valid = datasets.HistoGraph(data_path, '../../../../Set/Valid.txt', representation, normalization)
    data_test = datasets.HistoGraph(data_path, '../../../../Set/Test.txt', representation, normalization)

    return data_train, data_valid, data_test


def load_histograph_siamese(data_path, representation='adj', normalization=False):
    # Get data for train, validation and test
    data_train = datasets.HistoGraphSiamese(data_path, '../../../../Set/Train.txt', representation, normalization)
    data_valid = datasets.HistoGraphSiamese(data_path, '../../../../Set/Valid.txt', representation, normalization)
    data_test = datasets.HistoGraphSiamese(data_path, '../../../../Set/Test.txt', representation, normalization)

    return data_train, data_valid, data_test


def load_histograph_retrieval(data_path, representation='adj', normalization=False):
    data_train = datasets.HistoGraphRetrieval(data_path, '../../../02_GXL/02_PAR/01_Keypoint/2/', 'test.txt', 'keywords.txt', representation, normalization)
    data_valid = None
    data_test = datasets.HistoGraphRetrieval(data_path, '../../../02_GXL/02_PAR/01_Keypoint/2/', 'test.txt', 'keywords.txt', representation, normalization, test=True)
    return data_train, data_valid, data_test


def load_histograph_retrieval_siamese(data_path, representation='adj', normalization=False):
    data_train = datasets.HistoGraphRetrievalSiamese(data_path, '../../../02_GXL/02_PAR/01_Keypoint/2/', 'train.txt', representation, normalization)
    data_valid = datasets.HistoGraphRetrievalSiamese(data_path, '../../../02_GXL/02_PAR/01_Keypoint/2/', 'valid.txt', representation, normalization)
    data_test = datasets.HistoGraphRetrievalSiamese(data_path, '../../../02_GXL/02_PAR/01_Keypoint/2/', 'test.txt', representation, normalization)
    return data_train, data_valid, data_test


def collate_fn_multiple_size(batch):
    n_batch = len(batch)
    
    g_size = torch.LongTensor([x[0].size(0) for x in batch])
    graph_size = torch.LongTensor([[x[0].size(0), x[0].size(1), x[1].size(2)] for x in batch])
    sz, _ = graph_size.max(dim=0)
    sz = sz.squeeze()
    n_labels = torch.zeros(n_batch, sz[0], sz[1])

    am = torch.zeros(n_batch, sz[0], sz[0], sz[2])
    targets = torch.from_numpy(np.array([x[2] for x in batch])).long()

    for i in range(n_batch):
        # Node Features
        n_labels[i, :g_size[i], :] = batch[i][0]

        # Adjacency matrix
        am[i, :g_size[i], :g_size[i], :] = batch[i][1]

    return n_labels, am, g_size, targets


def collate_fn_multiple_size_siamese(batch):
    n_batch = len(batch)

    g_size1 = torch.LongTensor([x[0].size(0) for x in batch])
    g_size2 = torch.LongTensor([x[2].size(0) for x in batch])

    graph_size1 = torch.LongTensor([[x[0].size(0), x[0].size(1), x[1].size(2)] for x in batch])
    graph_size2 = torch.LongTensor([[x[2].size(0), x[2].size(1), x[3].size(2)] for x in batch])

    sz1, _ = graph_size1.max(dim=0)
    sz2, _ = graph_size2.max(dim=0)

    n_labels1 = torch.zeros(n_batch, sz1[0], sz1[1])
    n_labels2 = torch.zeros(n_batch, sz2[0], sz2[1])

    am1 = torch.zeros(n_batch, sz1[0], sz1[0], sz1[2])
    am2 = torch.zeros(n_batch, sz2[0], sz2[0], sz2[2])

    targets = torch.cat([x[-1] for x in batch])

    for i in range(n_batch):
        # Node Features
        n_labels1[i, :g_size1[i], :] = batch[i][0]
        n_labels2[i, :g_size2[i], :] = batch[i][2]

        # Adjacency matrix
        am1[i, :g_size1[i], :g_size1[i], :] = batch[i][1]
        am2[i, :g_size2[i], :g_size2[i], :] = batch[i][3]

    return n_labels1, am1, g_size1, n_labels2, am2, g_size2, targets
