# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Without training, compute the Hausdorff Distance between graph nodes.

"""

# Python modules
import torch
import time
from torch.autograd.variable import Variable

# Own modules
from options import Options
import datasets
from LogMetric import AverageMeter
from utils import knn
import GraphEditDistance

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


def train(train_loader, net, cuda, evaluation):
    batch_time = AverageMeter()
    acc = AverageMeter()

    eval_k = (1, 3, 5)

    end = time.time()

    for i, (h1, _, g_size1, target1) in enumerate(train_loader):
        # Prepare input data
        if cuda:
            h1, g_size1, target1 = h1.cuda(), g_size1.cuda(), target1.cuda()
        h1, target1 = Variable(h1), Variable(target1)

        D_aux = []
        T_aux = []
        for j, (h2, _, g_size2, target2) in enumerate(train_loader):
            # Prepare input data
            if cuda:
                h2, g_size2, target2 = h2.cuda(), g_size2.cuda(), target2.cuda()
            h2, target2 = Variable(h2), Variable(target2)

            d = net(h1, g_size1, h2, g_size2)

            # avoid classification as himself
            if i == j:
                d = d + torch.cat([d.max()]*d.size(0)).diag()

            D_aux.append(d)
            T_aux.append(target2)

        D = torch.cat(D_aux, 1)
        T = torch.cat(T_aux, 0)

        bacc = evaluation(D, target1, T, eval_k)

        # Measure elapsed time
        acc.update(bacc.data, h1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    print('Train distance:')
    for i in range(len(eval_k)):
        print('\t* {k}-NN; Average Acc {acc:.3f}; Avg Time x Batch {b_time.avg:.3f}'.format(k=eval_k[i], acc=acc.avg[i],
                                                                                            b_time=batch_time))

    return acc


def test(test_loader, train_loader, net, cuda, evaluation):
    batch_time = AverageMeter()
    acc = AverageMeter()

    eval_k = (1, 3, 5)

    end = time.time()

    for i, (h1, _, g_size1, target1) in enumerate(test_loader):
        # Prepare input data
        if cuda:
            h1, g_size1, target1 = h1.cuda(), g_size1.cuda(), target1.cuda()
        h1, target1 = Variable(h1), Variable(target1)

        D_aux = []
        T_aux = []
        for j, (h2, _, g_size2, target2) in enumerate(train_loader):
            # Prepare input data
            if cuda:
                h2, g_size2, target2 = h2.cuda(), g_size2.cuda(), target2.cuda()
            h2, target2 = Variable(h2), Variable(target2)

            d = net(h1, g_size1, h2, g_size2)

            D_aux.append(d)
            T_aux.append(target2)

        D = torch.cat(D_aux, 1)
        T = torch.cat(T_aux, 0)

        _, ind = D.sort()
        pred = T[ind[:, 0]]

        bacc = evaluation(D, target1, T, eval_k)

        # Measure elapsed time
        acc.update(bacc.data, h1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    print('Test distance:')
    for i in range(len(eval_k)):
        print('\t* {k}-NN; Average Acc {acc:.3f}; Avg Time x Batch {b_time.avg:.3f}'.format(k=eval_k[i], acc=acc.avg[i],
                                                                                            b_time=batch_time))
    return acc


def main():

    print('Prepare dataset')
    # Dataset
    data_train, data_valid, data_test = datasets.load_data(args.dataset, args.data_path, args.representation)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=datasets.collate_fn_multiple_size,
                                               batch_size=args.batch_size,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid,
                                               batch_size=args.batch_size, collate_fn=datasets.collate_fn_multiple_size,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.batch_size, collate_fn=datasets.collate_fn_multiple_size,
                                              num_workers=args.prefetch, pin_memory=True)

    print('Create model')
    net = GraphEditDistance.SoftHd()

    print('Loss & optimizer')
    evaluation = knn

    print('Check CUDA')
    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel **NOT TESTED**')
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print('\t* CUDA')
        net.cuda()

    print('Train')
    acc_train = train(train_loader, net, args.ngpu > 0, evaluation)

    print('Validation')
    acc_valid = test(valid_loader, train_loader, net, args.ngpu > 0, evaluation)

    # Evaluate best model in Test
    print('Test:')
    acc_test = test(test_loader, train_loader, net, args.ngpu > 0, evaluation)


if __name__ == '__main__':
    # Parse options
    args = Options().parse()

    # Check cuda
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()
    
    main()

