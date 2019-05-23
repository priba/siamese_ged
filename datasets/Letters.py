# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
import xml.etree.ElementTree as ET
import numpy as np
import datasets.data_utils as du
import itertools

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


class Letters(data.Dataset):
    def __init__(self, root_path, file_list, representation='adj', normalization=False):
        self.root = root_path
        self.file_list = file_list

        self.graphs, self.labels = getFileList(self.root + self.file_list)

        self.unique_labels = np.unique(self.labels)
        self.labels = [np.where(target == self.unique_labels)[0][0] for target in self.labels]

        self.representation = representation
        self.normalization = normalization

    def __getitem__(self, index):
        node_labels, am = create_graph_letter(self.root + self.graphs[index], representation=self.representation)
        target = self.labels[index]
        node_labels = torch.FloatTensor(node_labels)

        if self.normalization:
            node_labels = du.normalize_mean(node_labels)

        am = torch.FloatTensor(am)

        return node_labels, am, target
    
    def getId(self, index):
        return self.graphs[index]

    def __len__(self):
        return len(self.labels)

    def getTargetSize(self):
        return len(self.unique_labels)


class LettersSiamese(data.Dataset):
    def __init__(self, root_path, file_list, representation='adj', normalization=False):
        self.root = root_path
        self.file_list = file_list

        self.graphs, self.labels = getFileList(self.root + self.file_list)

        self.unique_labels = np.unique(self.labels)
        self.labels = [np.where(target == self.unique_labels)[0][0] for target in self.labels]

        self.pairs = list(itertools.permutations(range(len(self.labels)), 2))
        self.representation = representation
        self.normalization = normalization

        pair_label = np.array([self.labels[p[0]]==self.labels[p[1]] for p in self.pairs])
        self.weight = np.zeros(len(pair_label))
        self.weight[pair_label] = 1.0/pair_label.sum()
        self.weight[np.invert(pair_label)] = 1.0/np.invert(pair_label).sum()

    def __getitem__(self, index):
        ind = self.pairs[index]

        # Graph 1
        node_labels1, am1 = create_graph_letter(self.root + self.graphs[ind[0]], representation=self.representation)
        target1 = self.labels[ind[0]]
        node_labels1 = torch.FloatTensor(node_labels1)
        am1 = torch.FloatTensor(am1)

        if self.normalization:
            node_labels1 = du.normalize_mean(node_labels1)

        # Graph 2
        node_labels2, am2 = create_graph_letter(self.root + self.graphs[ind[1]], representation=self.representation)
        target2 = self.labels[ind[1]]
        node_labels2 = torch.FloatTensor(node_labels2)
        am2 = torch.FloatTensor(am2)

        if self.normalization:
            node_labels2 = du.normalize_mean(node_labels2)

        target = torch.FloatTensor([1.0]) if target1 == target2 else torch.FloatTensor([0.0])

        return node_labels1, am1, node_labels2, am2, target

    def __len__(self):
        return len(self.pairs)

    def getTargetSize(self):
        return len(self.unique_labels)

    def getWeights(self):
        return self.weight


def getFileList(file_path):
    elements = []
    classes = []
    tree = ET.parse(file_path)
    root = tree.getroot()
    for child in root:
        for sec_child in child:
            if sec_child.tag == 'print':
                elements += [sec_child.attrib['file']]
                classes += sec_child.attrib['class']
    return elements, classes


def create_graph_letter(file, representation='adj'):

    tree_gxl = ET.parse(file)
    root_gxl = tree_gxl.getroot()
    node_label = []
    node_id = []
    for node in root_gxl.iter('node'):
        node_id += [node.get('id')]
        for attr in node.iter('attr'):
            if (attr.get('name') == 'x'):
                x = float(attr.find('float').text)
            elif (attr.get('name') == 'y'):
                y = float(attr.find('float').text)
        node_label += [[x, y]]

    node_label = np.array(node_label)
    node_id = np.array(node_id)

    if representation=='adj':
        am = np.zeros((len(node_id), len(node_id), 1))
    else:
        am = np.zeros((len(node_id), len(node_id), 2))

    for edge in root_gxl.iter('edge'):
        s = np.where(np.array(node_id)==edge.get('from'))[0][0]
        t = np.where(np.array(node_id)==edge.get('to'))[0][0]

        if representation=='adj':
            am[s,t,:] = 1
            am[t,s,:] = 1
        else:
            dist = du.distance(node_label[s], node_label[t])
            am[s,t,:] = [dist, du.angle_between(node_label[s], node_label[t])]
            am[t,s,:] = [dist, du.angle_between(node_label[t], node_label[s])]

    return node_label, am

if __name__ == '__main__':
    from options import Options
    from load_data import load_data

    # Parse options
    args = Options().parse()

    # Dataset
    data_train, data_valid, data_test = load_data(args.dataset, args.data_path, args.representation,
                                                           args.normalization)

    nodes = 0
    edges = 0
    for node_labels, am, target in data_train:
        nodes = nodes + node_labels.size(0)
        if len(list(torch.nonzero(am).size())) > 0:
            edges = edges + torch.nonzero(am).size(0)/2.0

    for node_labels, am, target in data_valid:
        nodes = nodes + node_labels.size(0)
        if len(list(torch.nonzero(am).size()))>0:
            edges = edges + torch.nonzero(am).size(0)/2.0

    for node_labels, am, target in data_test:
        nodes = nodes + node_labels.size(0)
        if len(list(torch.nonzero(am).size())) > 0:
            edges = edges + torch.nonzero(am).size(0)/2.0

    nodes = nodes/(len(data_train)+len(data_valid)+len(data_test)+0.0)
    edges = edges / (len(data_train) + len(data_valid) + len(data_test)+0.0)

    print('Nodes: ' + str(nodes))
    print('Edges: ' + str(edges))
