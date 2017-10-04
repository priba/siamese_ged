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

