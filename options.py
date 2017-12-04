# -*- coding: utf-8 -*-
"""
Parse input arguments
"""

import argparse

__author__ = 'Pau Riba'
__email__ = 'priba@cvc.uab.cat'


class Options():

    def __init__(self):
        # MODEL SETTINGS
        parser = argparse.ArgumentParser(description='Train a Neural Message passing approach for graph retrieval',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # Positional arguments
        parser.add_argument('data_path', type=str, help='Root for the dataset.')
        parser.add_argument('dataset', type=str, choices=['letters', 'histograph'],
                            help='Choose between letters.')
        # Model parameters
        parser.add_argument('--nlayers', type=int, default=2, help='Message passing + Update layers.')
        parser.add_argument('--distance', type=str, default='SoftHd', help='Graph distance used.', choices=['Hd', 'SoftHd'])
        parser.add_argument('--representation', type=str, default='adj', help='Adjacency matrix representation.', choices=['adj', 'feat'])
        parser.add_argument('--normalization', action='store_true', default=False, help='Normalize data nodes.')
        parser.add_argument('--hidden_size', type=int, default=64, help='Hidden state size.')
        parser.add_argument('--write', type=str, default=None, help='If not None, write the whole dataset into memory before the Readout.')
        # Optimization options
        parser.add_argument('--epochs', '-e', type=int, default=1, help='Number of epochs to train.')
        parser.add_argument('--batch_size', '-b', type=int, default=375, help='Batch size.')
        parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2, help='The Learning Rate.')
        parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
        parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
        parser.add_argument('--schedule', type=int, nargs='+', default=[150],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
        # Checkpoints
        parser.add_argument('--save', '-s', type=str, default=None, help='Folder to save checkpoints.')
        parser.add_argument('--load', '-l', type=str, default=None, help='Checkpoint path to resume / test.')
        parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
        # Acceleration
        parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU, 1 = CUDA, 1 < DataParallel')
        parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
        # i/o
        parser.add_argument('--log', type=str, default='./log/', help='Log folder.')
        parser.add_argument('--log-interval', type=int, default=0, metavar='N',
                            help='How many batches to wait before logging training status')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
