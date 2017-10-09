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
from utils import accuracy
import GraphEditDistance

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


def train(train_loader, net, cuda, evaluation):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()

    for i, (h1, _, _, target1) in enumerate(train_loader):
        # Prepare input data
        if cuda:
            h1, target1 = h1.cuda(), target1.cuda()
        h1, target1 = Variable(h1), Variable(target1)

        for j, (h2, _, _, target2) in enumerate(train_loader):
            # Prepare input data
            if cuda:
                h2, target2 = h2.cuda(), target2.cuda()
            h2, target2 = Variable(h2), Variable(target2)

            net(h1, h2)

        # Measure data loading time
        data_time.update(time.time() - end)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: [{0}] Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(epoch, loss=losses, b_time=batch_time))

    return losses


def test(test_loader, train_loader, net, cuda, criterion, evaluation):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc = AverageMeter()

    end = time.time()

    for i, (h1, _, _, target1) in enumerate(test_loader):
        # Prepare input data
        if cuda:
            h1, target1 = h1.cuda(), target1.cuda()
        h1, target1 = Variable(h1, volatile=True), Variable(target1, volatile=True)

        for j, (h2, _, _, target2) in enumerate(test_loader):
            # Prepare input data
            if cuda:
                h2, target2 = h2.cuda(), target2.cuda()
            h2, target2 = Variable(h2, volatile=True), Variable(target2, volatile=True)
            
            net(h1, h2)

        # Measure data loading time
        data_time.update(time.time() - end)

        bacc = evaluation(output, target)
        
        acc.update(bacc[0].data[0], h.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Test: Average Loss {loss.avg:.3f}; Average ACC {acc.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(loss=losses, acc=acc, b_time=batch_time))

    return losses, acc


def main():

    print('Prepare dataset')
    # Dataset
    data_train, data_valid, data_test = datasets.load_data(args.dataset, args.data_path)

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
    net = GraphEditDistance.Hd()

    print('Loss & optimizer')
    evaluation = accuracy

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

