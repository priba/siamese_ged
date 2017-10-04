# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Train a siamese Network

Siamese network. Uses the same net until the Readout for both graph and compare their representation after the Readout to obtain a Loss.
"""

# Python modules
import torch
import time
from torch.autograd.variable import Variable

# Own modules
from option import Options
import datasets
from LogMetric import AverageMeter, Logger

if __name__ == '__main__':
    # Parse options
    args = Options().parse()

    # Check cuda
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()

    if not args.test:
        print('Initialize logger')
        log_dir = args.log + '{}_run-batchSize_{}/' \
                .format(len(glob.glob(args.log + '*_run-batchSize_{}'.format(args.batch_size))),args.batch_size)

        # Create Logger
        logger = Logger(log_dir, force=True)

    main()

