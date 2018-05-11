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

```
    $ python train_siamese_distance.py -h
    usage: train_siamese_distance.py [-h] [--nlayers NLAYERS]
                                     [--distance {Hd,SoftHd}]
                                     [--representation {adj,feat}]
                                     [--normalization] [--hidden_size HIDDEN_SIZE]
                                     [--write WRITE] [--epochs EPOCHS]
                                     [--batch_size BATCH_SIZE]
                                     [--learning_rate LEARNING_RATE]
                                     [--momentum MOMENTUM] [--decay DECAY]
                                     [--schedule SCHEDULE [SCHEDULE ...]]
                                     [--gamma GAMMA] [--save SAVE] [--load LOAD]
                                     [--test] [--ngpu NGPU] [--prefetch PREFETCH]
                                     [--log LOG] [--log-interval N]
                                     data_path
                                     {letters,histograph,histographretrieval}
```

Test a trained model

```
    $ python train_siamese_distance.py $DATA_PATH histograph -t -l ./trained_models/HistoGraph/01_Keypoint/checkpoint.pth --distance SoftHd -b 128 --hidden_size 64 --nlayers 3 --representation feat
    Test distance:
        * 1-NN; Average Acc 87.413; Avg Time x Batch 3.246
        * 3-NN; Average Acc 87.413; Avg Time x Batch 3.246
        * 5-NN; Average Acc 83.217; Avg Time x Batch 3.246
```

<img src="https://github.com/priba/siamese_ged/blob/master/readme_plots/pipeline.png" width="400">

## Bibliography
- [1] Gilmer *et al.*, [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf), arXiv, 2017.

## Authors

* [Pau Riba](http://www.cvc.uab.es/people/priba/) ([@priba](https://github.com/priba))
