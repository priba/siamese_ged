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

# Own modules
from options import Options
import datasets
from LogMetric import AverageMeter, Logger
from utils import save_checkpoint, load_checkpoint

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


def main():

    print('Prepare dataset')
    # Dataset
    data_train, data_valid, data_test = datasets.load_data(args.dataset, args.data_path)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid,
                                               batch_size=args.batch_size,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.batch_size,
                                              num_workers=args.prefetch, pin_memory=True)

    print('Create model')
    net = None

    print('Loss & optimizer')
    criterion = torch.nn.Loss()
    evaluation = None
    optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)
    
    print('Check CUDA')
    if ars.cuda and args.ngpu > 1:
        print('\t* Data Parallel **NOT TESTED**')
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print('\t* CUDA')
        net.cuda()

    start_epoch = 0
    
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
            logger.add_scalar('add_valid', cer_valid.avg)
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

