#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    HaussdorfEditdistance.py: Computes an aproximated Graph Edit Distance.

    * Bibliography: Fischer et al. (2015) "Approximation of graph edit distance based on Hausdorff matching."

    Usage:
"""

from __future__ import print_function

import torch
import torch.nn as nn

# Own modules

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat" 


class Hed(nn.Module):

    # Constructor
    def __init__(self, args={}):
        super(Hed, self).__init__()
        self.args = args
        self.node_in = args['node_in']
        self.edge_in = args['edge_in']

        self.node_insertion = nn.Parameter(torch.randn(1, self.node_in))
        self.node_deletion = nn.Parameter(torch.randn(1, self.node_in))
        self.node_substitution = nn.Parameter(torch.randn(self.node_in, self.node_in))

        self.edge_insertion = nn.Parameter(torch.randn(1, self.edge_in))
        self.edge_deletion = nn.Parameter(torch.randn(1, self.edge_in))
        self.edge_substitution = nn.Parameter(torch.randn(self.edge_in, self.edge_in))
        
    def forward(self, v1, am1, sz1, v2, am2, sz2):
        batch_sz = v1.size(0)
        # Deletion G1
        # Nodes:
        d1 = self.deletion_distance(v1, am1, sz1) 
        # Edges

        # Insertion G2
        d2 = self.insertion_distance(v2, am2, sz2) 

        # Substitution G1-G2
        ce = self.hec(am1, sz1, am2, sz2)
        ce = torch.max(, ce)
        # Hausdorff distance
        d = d1.sum() + d2.sum()
        return d

    def deletion_distance(self, v, am, sz):
        # Node
        v_view = v.view(-1, v.size(-1), 1)
        del_view = self.node_deletion.unsqueeze(0).expand(v_view.size(0), 1, self.node_in)
        d = del_view.bmm(v_view)
        d = d.view(v.size(0), -1)

        # Edge
        am_view = am.view(-1, am.size(-1), 1)
        del_edg_view = self.edge_deletion.unsqueeze(0).expand_as(v_view.size(0), 1, self.edge_in)
        d_edge = del_edg_view.bmm(am_view)
        d_edge = d_edge/2.0
        d_edge = d_edge.view(am.size(0), am.size(1), am.size(2))
        d_edge = d_edge.sum(2)

        return d + d_edge

    def insertion_distance(self, v, am, sz):
        # Node
        v_view = v.view(-1, v.size(-1), 1)
        ins_view = self.node_insertion.unsqueeze(0).expand(v_view.size(0), 1, self.node_in)
        d = ins_view.bmm(v_view)
        d = d.view(v.size(0), -1)

        # Edge
        am_view = am.view(-1, am.size(-1), 1)
        ins_edg_view = self.edge_insertion.unsqueeze(0).expand_as(v_view.size(0), 1, self.edge_in)
        d_edge = ins_edg_view.bmm(am_view)
        d_edge = d_edge/2.0
        d_edge = d_edge.view(am.size(0), am.size(1), am.size(2))
        d_edge = d_edge.sum(2)

        return d + d_edge

    def hec(self, am1, sz1, am2, sz2):
        # Deletion
        c1 = self.edge_deletion_distance(am1, sz1)

        # Insertion
        c2 = self.edge_insertion_distance(am2, sz2)

        # Substitution
        c = self.edge_substitution_distance(am1, sz1, am2, sz2)    
        c1 = torch.min(c.min(2).view_as(c1), c1)
        c2 = torch.min(c.min(2).view_as(c2), c2)
        
        # Cost
        c = c1.sum(2) + c2.sum(2)

        return c

    def edge_deletion_distance(self, am, sz):
        # Edge
        am_view = am.view(-1, am.size(-1), 1)
        del_edg_view = self.edge_deletion.unsqueeze(0).expand_as(am_view.size(0), 1, self.edge_in)
        d_edge = del_edg_view.bmm(am_view)
        d_edge = d_edge.view(am.size(0), am.size(1), am.size(2))

        return d_edge

    def edge_insertion_distance(self, am, sz):
        # Edge
        am_view = am.view(-1, am.size(-1), 1)
        ins_edg_view = self.edge_insertion.unsqueeze(0).expand_as(am_view.size(0), 1, self.edge_in)
        d_edge = ins_edg_view.bmm(am_view)
        d_edge = d_edge.view(am.size(0), am.size(1), am.size(2))

        return d_edge

    def edge_substitution_distance(self, am1, sz1, am2, sz2): 
        am1_view = am1.view(am1.size(0), -1, 1, am1.size(-1), 1)
        am2_view = am2.view(am2.size(0), -1, 1, 1, am2.size(-1))

        am1_view = am1_view.expand( am1_view.size(0), am1_view.size(1),
                                    am2_view.size(1), am1_veiw.size(3)
                                    am1_view.size(4))
        am2_view = am2_view.expand( am2_view.size(0), am2_view.size(1),
                                    am1_view.size(1), am2_veiw.size(3)
                                    am2_view.size(4)).transpose(1, 2)

        am1_view = am1_view.view( -1, am1.size(-1), 1 )
        am2_view = am2_view.view( -1, 1, am2.size(-1) )
        
        sub_edg_view = self.edge_substitution.unsqueeze(0).expand_as(am1_view.size(0), self.edge_in, self.edge_in)

        d_edge = sub_edg_view.bmm(am1_view)
        d_edge = am2_view.bmm(d_edge)

        d_edge = d_edge/2.0

        d_edge = d_edge.view( am1.size(0), am1.size(1)*am1.size(2), am2.size(1)*am2.size(2) )

        return d_edge

    def L_graph(self, c1, sz1, c2, sz2):
        mask = sz1 > sz2
        return mask.long() * c1.min(1) + (1-mask.long) * c2.min(1)

