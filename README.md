# Learning Graph Distances with Message PassingNeural Networks

Siamese Neural message passing for graph retrieval implementation.

* Pau Riba, Andreas Fischer, Josep Lladós and Alicia Fornes, Learning Graph Distances with Message PassingNeural Networks, ICPR2018 (Submitted)

## Installation

    $ pip install -r requirements.txt

## Methodology.

### Graph Distance

Compute the distance using raw graphs without training:

    $ python no_train_hausdorff.py

### Learn Representation

Trains a classifier. In test time discards the readout and uses a graph distance.

    $ python train_learn_representation.py

### Siamese NMP Network

Siamese network. Maps the graph in an Euclidean space and computes the distance.

    $ python train_siamese_net.py


### Siamese NMP Distance Network

Siamese network. Enrich the graph with node features and computes a Hausdorff Distance

    $ python train_siamese_distance.py        

<img src="https://github.com/priba/siamese_ged/blob/master/readme_plots/pipeline.png" width="400">

## Bibliography
- [1] Gilmer *et al.*, [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf), arXiv, 2017.

## Authors

* [Pau Riba](http://www.cvc.uab.es/people/priba/) ([@priba](https://github.com/priba))
