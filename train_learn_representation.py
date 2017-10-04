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

