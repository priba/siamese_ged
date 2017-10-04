# -*- coding: utf-8 -*-
import torch.utils.data as data
import xml.etree.ElementTree as ET

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"

class Letters(data.Dataset):
    def __init__(self, root_path, file_list)
        self.root = root_path
        self.file_list = file_list

        self.graphs, self.labels = self.getFileList()

    def __getitem__(self, index):
        target = self.classes[index]
        return _, target

    def getFileList(self):
        elements = []
        classes = []
        tree = ET.parse(self.root + self.file_list)
        root = tree.getroot()
        for child in root:
            for sec_child in child:
                if sec_child.tag=='print':
                    elements += [sec_child.attrib['file']]
                    classes += sec_child.attrib['class']
        return elements, classes

    def create_graph_letter(self, file):

        tree_gxl = ET.parse(file)
        root_gxl = tree_gxl.getroot()
        vl = []
        for node in root_gxl.iter('node'):
            for attr in node.iter('attr'):
                if (attr.get('name') == 'x'):
                    x = float(attr.find('float').text)
                elif (attr.get('name') == 'y'):
                    y = float(attr.find('float').text)
            vl += [[x, y]]
        
        for edge in root_gxl.iter('edge'):
            s = int(edge.get('from').split('_')[1])
            t = int(edge.get('to').split('_')[1])

        for i in range(len(vl)):
            np.array(vl[i][:2])

        return _ 
