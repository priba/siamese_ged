# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Siamese Neural Message Passing distance.

Learn a Graph Edit Distance training jointly with a Neural message passing network.
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
from utils import save_checkpoint, load_checkpoint, siamese_accuracy, knn, meanAveragePrecision, write_gxl
import models
import GraphEditDistance
import LossFunction

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


def train(train_loader, net, distance, optimizer, cuda, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    net.train()
    distance.train()

    end = time.time()

    for i, (h1, am1, g_size1, h2, am2, g_size2, target) in enumerate(train_loader):
        # Prepare input data
        if cuda:
            h1, am1, g_size1 = h1.cuda(), am1.cuda(), g_size1.cuda()
            h2, am2, g_size2 = h2.cuda(), am2.cuda(), g_size2.cuda()
            target = target.cuda()
        h1, am1, g_size1 = Variable(h1), Variable(am1), Variable(g_size1)
        h2, am2, g_size2 = Variable(h2), Variable(am2), Variable(g_size2)
        target = Variable(target)


        # Measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        # Compute features
        output1 = net(h1, am1, g_size1, output='nodes')
        output2 = net(h2, am2, g_size2, output='nodes')

        # Create a mask for nodes
        node_mask2 = torch.arange(0, h2.size(1)).unsqueeze(0).unsqueeze(-1).expand(h2.size(0),
                                                                                   h2.size(1),
                                                                                   output1.size(2)).long()
        node_mask1 = torch.arange(0, h1.size(1)).unsqueeze(0).unsqueeze(-1).expand(h1.size(0),
                                                                                   h1.size(1),
                                                                                   output1.size(2)).long()

        if h1.is_cuda:
            node_mask1 = node_mask1.cuda()
            node_mask2 = node_mask2.cuda()

        node_mask1 = Variable(node_mask1)
        node_mask2 = Variable(node_mask2)

        node_mask1 = (node_mask1 >= g_size1.unsqueeze(-1).unsqueeze(-1).expand_as(node_mask1))
        node_mask2 = (node_mask2 >= g_size2.unsqueeze(-1).unsqueeze(-1).expand_as(node_mask2))


        output1.register_hook(lambda grad: grad.masked_fill_(node_mask1, 0))
        output2.register_hook(lambda grad: grad.masked_fill_(node_mask2,0))
       
        output = distance(output1, am1, g_size1, output2, am2, g_size2)

        loss = criterion(output, target)

        # Logs
        losses.update(loss.data[0], h1.size(0))

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: [{0}] Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(epoch, loss=losses, b_time=batch_time))

    return losses


def validation(test_loader, net, distance, cuda, criterion, evaluation):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    net.eval()
    distance.eval()

    end = time.time()

    for i, (h1, am1, g_size1, h2, am2, g_size2, target) in enumerate(test_loader):
        # Prepare input data
        if cuda:
            h1, am1, g_size1 = h1.cuda(), am1.cuda(), g_size1.cuda()
            h2, am2, g_size2 = h2.cuda(), am2.cuda(), g_size2.cuda()
            target = target.cuda()
        h1, am1, g_size1 = Variable(h1, volatile=True), Variable(am1, volatile=True), Variable(g_size1, volatile=True)
        h2, am2, g_size2 = Variable(h2, volatile=True), Variable(am2, volatile=True), Variable(g_size2, volatile=True)
        target = Variable(target, volatile=True)

        # Measure data loading time
        data_time.update(time.time() - end)

        # Compute features
        output1 = net(h1, am1, g_size1, output='nodes')
        output2 = net(h2, am2, g_size2, output='nodes')
        
        output = distance(output1, am1, g_size1, output2, am2, g_size2)

        loss = criterion(output, target)
        bacc = evaluation(output, target)

        # Logs
        losses.update(loss.data[0], h1.size(0))
        acc.update(bacc[0].data[0], h1.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Test: Average Loss {loss.avg:.3f}; Average Acc {acc.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(loss=losses, acc=acc, b_time=batch_time))

    return losses, acc


def test(test_loader, train_loader, net, distance, cuda, evaluation):
    batch_time = AverageMeter()
    acc = AverageMeter()

    eval_k = (1, 3, 5)

    # switch to train mode
    net.eval()
    distance.eval()

    end = time.time()

    for i, (h1, am1, g_size1, target1) in enumerate(test_loader):
        # Prepare input data
        if cuda:
            h1, am1, g_size1, target1 = h1.cuda(), am1.cuda(), g_size1.cuda(), target1.cuda()
        h1, am1, g_size1, target1 = Variable(h1, volatile=True), Variable(am1, volatile=True), Variable(g_size1, volatile=True), Variable(target1, volatile=True)

        # Compute features
        output1 = net(h1, am1, g_size1, output='nodes')

        D_aux = []
        T_aux = []
        for j, (h2, am2, g_size2, target2) in enumerate(train_loader):
            # Prepare input data
            if cuda:
                h2, am2, g_size2, target2 = h2.cuda(), am2.cuda(), g_size2.cuda(), target2.cuda()
            h2, am2, g_size2, target2 = Variable(h2, volatile=True), Variable(am2, volatile=True), Variable(g_size2, volatile=True), Variable(target2, volatile=True)

            # Compute features
            output2 = net(h2, am2, g_size2, output='nodes')

            # Expand test sample to make all the pairs with the train
            dist = distance(output1.expand(h2.size(0), output1.size(1), output1.size(2)),
                    am1.expand(am2.size(0), am1.size(1), am1.size(2), am1.size(3)), g_size1.expand(g_size2.size(0)), output2, am2, g_size2)

            D_aux.append(dist)
            T_aux.append(target2)

        D = torch.cat(D_aux)
        train_target = torch.cat(T_aux, 0)

        if evaluation.__name__ == 'knn':
            bacc = evaluation(D, target1, train_target, k=eval_k)
        else:
            _, i = torch.min(D, 0)
            D = torch.cat([D[:int(i)], D[int(i) + 1:]])
            train_target = torch.cat([train_target[:int(i)], train_target[int(i) + 1:]])
            bacc = evaluation(D, target1, train_target)

        # Measure elapsed time
        acc.update(bacc, h1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    print('Test distance:')
    if evaluation.__name__ == 'knn':
        for i in range(len(eval_k)):
            print('\t* {k}-NN; Average Acc {acc:.3f}; Avg Time x Batch {b_time.avg:.3f}'.format(k=eval_k[i], acc=acc.avg[i], b_time=batch_time))
    else:
        print('\t* MAP {acc:.3f}; Avg Time x Batch {b_time.avg:.3f}'.format(acc=acc.avg, b_time=batch_time))
    return acc


def main():

    print('Prepare dataset')
    # Dataset
    data_train, data_valid, data_test = datasets.load_data(args.dataset, args.data_path, args.representation, args.normalization, siamese=True)


    if args.dataset == 'letters':
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(data_train.getWeights(), 2*15*50*(50-1), replacement=True)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(torch.multinomial(torch.DoubleTensor(data_valid.getWeights()), 50*(50-1), replacement=False))
    elif args.dataset == 'histograph':
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(data_train.getWeights(), 3*(3-1)*30*2, replacement=True)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
            torch.multinomial(torch.DoubleTensor(data_valid.getWeights()), 3*(3-1)*30*2, replacement=False))
    else:
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(data_train.getWeights(), 3 * (3 - 1) * 30 * 2,
                                                                       replacement=True)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
            torch.multinomial(torch.DoubleTensor(data_valid.getWeights()), 3 * (3 - 1) * 30 * 2, replacement=False))

    # Data Loader
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=datasets.collate_fn_multiple_size_siamese,
                                               batch_size=args.batch_size, sampler=train_sampler,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, sampler=valid_sampler,
                                               batch_size=args.batch_size, collate_fn=datasets.collate_fn_multiple_size_siamese,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.batch_size, collate_fn=datasets.collate_fn_multiple_size_siamese,
                                              num_workers=args.prefetch, pin_memory=True)

    print('Create model')
    if args.representation=='adj':
        print('\t* Discrete Edges')
        net = models.MpnnGGNN(in_size=2, e=[1], hidden_state_size=args.hidden_size, message_size=args.hidden_size, n_layers=args.nlayers, discrete_edge=True, out_type='regression', target_size=None)
    elif args.representation=='feat':
        print('\t* Feature Edges')
        net = models.MpnnGGNN(in_size=2, e=2, hidden_state_size=args.hidden_size, message_size=args.hidden_size, n_layers=args.nlayers, discrete_edge=False, out_type='regression', target_size=None)
    else:
        raise NameError('Representation ' + args.representation + ' not implemented!')

    print('Distance')
    if args.distance=='Hd':
        print('\t* Hausdorff Distance')
        distance = GraphEditDistance.Hd()
    elif args.distance=='SoftHd':
        print('\t* Soft Hausdorff Distance')
        distance = GraphEditDistance.SoftHd()
    elif args.distance=='Hed':
        print('\t* Hausdorff Edit Distance')
        distance = GraphEditDistance.Hed(args={'node_in': 64, 'edge_in': 2})
    else:
        raise NameError('Distance ' + args.distance + ' not implemented!')

    print('Loss & optimizer')
    criterion = LossFunction.ContrastiveLoss()
    evaluation = siamese_accuracy
    optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)
    
    print('Check CUDA')
    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel **NOT TESTED**')
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
        distance = torch.nn.DataParallel(distance, device_ids=list(range(args.ngpu)))
        
    if args.cuda:
        print('\t* CUDA')
        net = net.cuda()
        distance = distance.cuda()

    start_epoch = 0
    best_acc = 0
    if args.load is not None:
        print('Loading model')
        checkpoint = load_checkpoint(args.load)
        net.load_state_dict(checkpoint['state_dict'])
        distance.load_state_dict(checkpoint['distance_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

    if not args.test:

        print('Training loop')
        # Main loop
        for epoch in range(start_epoch, args.epochs):
            # update the optimizer learning rate
            adjust_learning_rate(optimizer, epoch)

            loss_train = train(train_loader, net, distance, optimizer, args.ngpu > 0, criterion, epoch)
            loss_valid, acc_valid = validation(valid_loader, net, distance, args.ngpu > 0, criterion, evaluation)

            # Save model
            if args.save is not None:
                if acc_valid.avg > best_acc:
                    best_acc = acc_valid.avg
                    save_checkpoint({'epoch': epoch + 1, 'state_dict': net.state_dict(), 'distance_state_dict': distance.state_dict(), 'best_acc': best_acc},
                                    directory=args.save, file_name='checkpoint')

            # Logger step
            # Scalars
            logger.add_scalar('loss_train', loss_train.avg)
            logger.add_scalar('loss_valid', loss_valid.avg)
            logger.add_scalar('acc_valid', acc_valid.avg)
            logger.add_scalar('learning_rate', args.learning_rate)

            logger.step()

        # Load Best model to evaluate in test if we are saving it in a checkpoint
        if args.save is not None:
            print('Loading best model to test')
            best_model_file = os.path.join(args.save, 'checkpoint.pth')
            checkpoint = load_checkpoint(best_model_file)
            net.load_state_dict(checkpoint['state_dict'])
            distance.load_state_dict(checkpoint['distance_state_dict'])

    # Evaluate best model in Test
    print('Test:')
    #loss_test, acc_test = validation(test_loader, net, distance, args.ngpu > 0, criterion, evaluation)

    # Dataset not siamese for test
    data_train, data_valid, data_test = datasets.load_data(args.dataset, args.data_path, args.representation, args.normalization)
    # Data Loader
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=datasets.collate_fn_multiple_size,
                                               batch_size=args.batch_size,
                                               num_workers=args.prefetch, pin_memory=True)
    # The batchsize is given by the train_loader
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=1, collate_fn=datasets.collate_fn_multiple_size,
                                              num_workers=args.prefetch, pin_memory=True)
    print('Test k-NN classifier')
    if args.dataset =='histographretrieval':
        evaluation = meanAveragePrecision
    else:
        evaluation = knn
    acc_test_hd = test(test_loader, train_loader, net, distance, args.ngpu > 0, evaluation)

    if args.write is not None:
        if not os.path.exists(args.write):
            os.makedirs(args.write)

        directed = False if args.representation=='adj' else True
        
        # Train
        write_dataset(data_train, net, args.ngpu>0, directed)
        # Validation
        write_dataset(data_valid, net, args.ngpu>0, directed)
        # Test
        write_dataset(data_test, net, args.ngpu>0, directed)


def write_dataset(data, net, cuda, directed):
    for i in range(len(data)):
        v, am, _ = data[i]
        g_size = torch.LongTensor([v.size(0)])

        v, am = v.unsqueeze(0), am.unsqueeze(0)

        if cuda:
            v, am, g_size = v.cuda(), am.cuda(), g_size.cuda()
        v, am, g_size = Variable(v, volatile=True), Variable(am, volatile=True), Variable(g_size, volatile=True)
        # Compute features
        v = net(v, am, g_size, output='nodes')

        v, am = v.squeeze(0).data, am.squeeze(0).data

        write_gxl( v, am, args.write + data.getId(i), directed)


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

