# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Learn graph representation.

Learns a classifier using a Neural Message Passing network. Drops the Readout layer and uses the learned representation in a Graph Edit distance framework.
"""

# Python modules
import torch
import time
from torch.autograd.variable import Variable
import glob
import os

# Own modules
from options import Options
import datasets
from LogMetric import AverageMeter, Logger
from utils import save_checkpoint, load_checkpoint, accuracy
import models

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


def train(train_loader, net, optimizer, cuda, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()

    for i, (h, am, g_size, target) in enumerate(train_loader):
        # Prepare input data
        if cuda:
            h, am, g_size, target = h.cuda(), am.cuda(), g_size.cuda(), target.cuda()
        h, am, g_size, target = Variable(h), Variable(am), Variable(g_size), Variable(target)

        # Measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        # Compute features
        output = net(h, am, g_size)

        loss = criterion(output, target)

        # Logs
        losses.update(loss.data[0], h.size(0))

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: [{0}] Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(epoch, loss=losses, b_time=batch_time))

    return losses


def test(test_loader, net, cuda, criterion, evaluation):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to eval mode
    net.eval()

    end = time.time()

    for i, (h, am, g_size, target) in enumerate(test_loader):
        # Prepare input data
        if cuda:
            h, am, target = h.cuda(), am.cuda(), target.cuda()
        h, am, target = Variable(h, volatile=True), Variable(am, volatile=True), Variable(target, volatile=True)

        # Measure data loading time
        data_time.update(time.time() - end)

        # Compute features
        output = net(h, am, g_size)

        loss = criterion(output, target)
        bacc = evaluation(output, target)
        
        # Logs
        losses.update(loss.data[0], h.size(0))
        acc.update(bacc.data[0], h.size(0))

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
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid,
                                               batch_size=args.batch_size, collate_fn=datasets.collate_fn_multiple_size,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.batch_size, collate_fn=datasets.collate_fn_multiple_size,
                                              num_workers=args.prefetch, pin_memory=True)

    print('Create model')
    net = models.MpnnGGNN(in_size=2, e=[1], hidden_state_size=64, message_size=64, n_layers=2, target_size=data_train.getTargetSize())

    print('Loss & optimizer')
    criterion = torch.nn.NLLLoss()
    evaluation = accuracy
    optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)
    
    print('Check CUDA')
    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel **NOT TESTED**')
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print('\t* CUDA')
        net.cuda()

    start_epoch = 0
    best_acc = 0
    if args.load is not None:
        print('Loading model')
        checkpoint = load_checkpoint(args.load)
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

    if not args.test:

        print('Training loop')
        # Main loop
        for epoch in range(start_epoch, args.epochs):
            # update the optimizer learning rate
            adjust_learning_rate(optimizer, epoch)

            loss_train = train(train_loader, net, optimizer, args.ngpu > 0, criterion, epoch)
            loss_valid, acc_valid = test(valid_loader, net, args.ngpu > 0, criterion, evaluation)

            # Save model
            if args.save is not None:
                if acc_valid.avg < best_acc:
                    best_acc = acc_valid.avg
                    save_checkpoint({'epoch': epoch + 1, 'state_dict': net.state_dict(), 'best_acc': best_acc},
                                    directory=args.save, file_name='checkpoint')

            # Logger step
            # Scalars
            logger.add_scalar('loss_train', loss_train.avg)
            logger.add_scalar('loss_valid', loss_valid.avg)
            logger.add_scalar('add_valid', acc_valid.avg)
            logger.add_scalar('learning_rate', args.learning_rate)

            logger.step()

        # Load Best model to evaluate in test if we are saving it in a checkpoint
        if args.save is not None:
            print('Loading best model to test')
            best_model_file = os.path.join(args.save, 'checkpoint.pth')
            checkpoint = load_checkpoint(best_model_file)
            net.load_state_dict(checkpoint['state_dict'])

    # Evaluate best model in Test
    print('Test:')
    loss_test, acc_test = test(test_loader, net, args.ngpu > 0, criterion, evaluation, char_enc)

def adjust_learning_rate(optimizer, epoch):
    """Updates the learning rate given an schedule and a gamma parameter.
    """
    if epoch in args.schedule:
        args.learning_rate *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate

if __name__ == '__main__':
    # Parse options
    args = Options().parse()

    # Check cuda
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()
    
    # Check Test and load
    if args.test and args.load is None:
        raise Exception('Cannot test withoud loading a model.')

    if not args.test:
        print('Initialize logger')
        log_dir = args.log + '{}_run-batchSize_{}/' \
                .format(len(glob.glob(args.log + '*_run-batchSize_{}'.format(args.batch_size))),args.batch_size)

        # Create Logger
        logger = Logger(log_dir, force=True)

    main()

