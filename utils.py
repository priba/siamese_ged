#-*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Pytorch useful tools.
"""

import torch
import os
import errno
import numpy as np

__author__ = 'Pau Riba'
__email__ = 'priba@cvc.uab.cat'


def save_checkpoint(state, directory, file_name):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, file_name + '.pth')
    torch.save(state, checkpoint_file)


def load_checkpoint(model_file):
    if os.path.isfile(model_file):
        print("=> loading model '{}'".format(model_file))
        checkpoint = torch.load(model_file)
        print("=> loaded model '{}' (epoch {}, acc {})".format(model_file, checkpoint['epoch'], checkpoint['best_acc']))
        return checkpoint
    else:
        print("=> no model found at '{}'".format(model_file))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), model_file)


def accuracy(output, target):
    return precision_at_k(output, target, topk=(1,))


def precision_at_k(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred = pred.type_as(target)
    target = target.type_as(pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def siamese_accuracy(output, target):
    batch_size = target.size(0)

    pred=(output > 0.5).float()
    correct = pred.eq(target).float()
    acc= 100.0*correct.sum()/batch_size
    return acc


def knn(D, target, train_target, k=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk=max(k)
    batch_size = target.size(0)
    _, pred = D.topk(maxk, dim=1, largest=False, sorted=True)
    pred = train_target[pred.view(-1)].view(batch_size, -1)
    pred = pred.type_as(target)

    res = []
    for ki in k:
        pred_k = nn_prediction(pred[:,:ki])

        pred_k = pred_k.squeeze()
        correct_k = pred_k.eq(target.data).float().sum()

        res.append(correct_k*(100.0 / batch_size))
    return torch.FloatTensor(res)

def nn_prediction(pred, axis=1):
    scores = np.unique(np.ravel(pred.data.cpu().numpy()))

    testshape = list(pred.size())
    testshape[axis] = 1

    mostfrequent = torch.zeros(testshape)
    mostindex = torch.zeros(testshape).long()
    oldcounts = torch.zeros(testshape)

    if pred.is_cuda:
        mostfrequent = mostfrequent.cuda()
        mostindex = mostindex.cuda()
        oldcounts = oldcounts.cuda()

    for score in scores:
        template = (pred == score).data
        counts = template.float().sum(axis, keepdim=True)
        _, ind = template.float().max(1)

        mostfrequent[(counts > oldcounts) | ((counts==oldcounts) & (ind.unsqueeze(1) < mostindex))] = score
        mostindex[(counts > oldcounts) | ((counts==oldcounts) & (ind.unsqueeze(1) < mostindex))] = ind

        oldcounts,_ = torch.max(torch.cat([oldcounts, counts],1),1, keepdim=True)

    return mostfrequent.long()
