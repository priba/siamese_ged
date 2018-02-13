# Learning Graph Distances with Message PassingNeural Networks

Neural message passing for graph retrieval implementation.

Pau Riba, Andreas Fischer, Josep Lladós and Alicia Fornes, Learning Graph Distances with Message PassingNeural Networks, ICPR2018 (Submitted)

## Installation

    $ pip install -r requirements.txt

## Methodology.
### Graph Distance
    $ python no_train_hausdorff.py

### Learn Representation
    
    $ python train_learn_representation.py

<img src="https://github.com/priba/nmp_ged/blob/master/readme_plots/learn_graph.png" width="800">

### Siamise NMP Network
    
    $ python train_siamese_net.py

<img src="https://github.com/priba/nmp_ged/blob/master/readme_plots/siamese_net.png" width="800">


### Siamise NMP Distance Network

    $ python train_siamese_distance.py        

<img src="https://github.com/priba/nmp_ged/blob/master/readme_plots/siamese_distance.png" width="800">

## Bibliography
- [1] Gilmer *et al.*, [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf), arXiv, 2017.

## Authors

* [Pau Riba](http://www.cvc.uab.es/people/priba/) ([@priba](https://github.com/priba))
