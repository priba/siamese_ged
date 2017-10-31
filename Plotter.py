#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Plotter.py
    Visualization functions.
"""

import torch
from torch.autograd.variable import Variable
import networkx as nx
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import datasets
from options import Options
import models
from utils import load_checkpoint
import pdb

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


def plot_graph(v, am, outname):
    fig = plt.figure()

    A = (am.abs().sum(2)>0).cpu().numpy()
    g = nx.from_numpy_matrix(A)

    position = {k: v[k].cpu().numpy() for k in range(v.size(0))}

    center = np.mean(list(position.values()),axis=0)
    max_pos = np.max(np.abs(list(position.values())-center))

    nx.draw(g, pos=position)

    plt.ylim([center[1]-max_pos-0.5, center[1]+max_pos+0.5])
    plt.xlim([center[0]-max_pos-0.5, center[0]+max_pos+0.5])

    plt.savefig(outname)


def plot_dataset(data, net, cuda):
    for i in range(len(data)):
        print(data.getId(i))
        v, am, _ = data[i]
        g_size = torch.LongTensor([v.size(0)])

        plot_graph( v, am, 'original.png')

        v, am = v.unsqueeze(0), am.unsqueeze(0)    
        
        if cuda:
            v, am, g_size = v.cuda(), am.cuda(), g_size.cuda()
        v, am = Variable(v, volatile=True), Variable(am, volatile=True)
        # Compute features
        v = net(v, am, g_size, output='nodes')

        v, am = v.squeeze(0).data, am.squeeze(0).data
        
        plot_graph( v, am, 'processed.png')
        
        raw_input("Press Enter to continue...")

def main():

    print('Prepare dataset')
    # Dataset
    data_train, data_valid, data_test = datasets.load_data(args.dataset, args.data_path, args.representation, args.normalization)
    
    print('Create model')
    if args.representation=='adj':
        print('\t* Discrete Edges')
        net = models.MpnnGGNN(in_size=2, e=[1], hidden_state_size=args.hidden_size, message_size=args.hidden_size, n_layers=args.nlayers, discrete_edge=True, out_type='regression', target_size=data_train.getTargetSize())
    elif args.representation=='feat':
        print('\t* Feature Edges')
        net = models.MpnnGGNN(in_size=2, e=2, hidden_state_size=args.hidden_size, message_size=args.hidden_size, n_layers=args.nlayers, discrete_edge=False, out_type='regression', target_size=data_train.getTargetSize())
    else:
        raise NameError('Representation ' + args.representation + ' not implemented!')

    print('Check CUDA')
    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel **NOT TESTED**')
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
        
    if args.cuda:
        print('\t* CUDA')
        net = net.cuda()

    if args.load is not None:
        print('Loading model')
        checkpoint = load_checkpoint(args.load)
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
    else:
        raise NameError('Load path must be set!') 

    # Train
    plot_dataset(data_train, net, args.ngpu>0)
    # Validation
    plot_dataset(data_valid, net, args.ngpu>0)
    # Test
    plot_dataset(data_test, net, args.ngpu>0)

        
if __name__ == '__main__':
    # Parse options
    args = Options().parse()

    # Check cuda
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()
    
    if args.load is None:
        raise Exception('Cannot plot without loading a model.')

    main()


