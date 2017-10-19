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


def test(test_loader, train_loader, net, cuda, evaluation):
    batch_time = AverageMeter()
    acc = AverageMeter()

    eval_k = (1, 3, 5)

    end = time.time()

    for i, (h1, am1, g_size1, target1) in enumerate(test_loader):
        # Prepare input data
        if cuda:
            h1, am1, g_size1, target1 = h1.cuda(), am1.cuda(), g_size1.cuda(), target1.cuda()
        h1, am1, target1 = Variable(h1), Variable(am1), Variable(target1)

        D_aux = []
        T_aux = []
        for j, (h2, am2, g_size2, target2) in enumerate(train_loader):
            # Prepare input data
            if cuda:
                h2, am2, g_size2, target2 = h2.cuda(), am2.cuda(), g_size2.cuda(), target2.cuda()
            h2, am2, target2 = Variable(h2), Variable(am2), Variable(target2)

            d = net(h1.expand(h2.size(0), h1.size(1), h1.size(2)),
                    am1.expand(am2.size(0), am1.size(1), am1.size(2), am1.size(2)),
                    g_size1.expand_as(g_size2), h2, am2, g_size2)

            D_aux.append(d)
            T_aux.append(target2)

        D = torch.cat(D_aux)
        train_target = torch.cat(T_aux, 0)

        bacc = evaluation(D, target1.expand_as(train_target), train_target, k=eval_k)

        # Measure elapsed time
        acc.update(bacc, h1.size(0))
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
    data_train, data_valid, data_test = datasets.load_data(args.dataset, args.data_path, args.representation, args.normalization)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=datasets.collate_fn_multiple_size,
                                               batch_size=args.batch_size,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid,
                                               batch_size=1, collate_fn=datasets.collate_fn_multiple_size,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=1, collate_fn=datasets.collate_fn_multiple_size,
                                              num_workers=args.prefetch, pin_memory=True)

    print('Create model')
    if args.distance=='SoftHd':
        net = GraphEditDistance.SoftHd()
    else:
        net = GraphEditDistance.Hd()

    print('Loss & optimizer')
    evaluation = knn

    print('Check CUDA')
    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel **NOT TESTED**')
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print('\t* CUDA')
        net = net.cuda()

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

