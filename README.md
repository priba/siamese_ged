# Learning Graph Distances with Message PassingNeural Networks

Siamese Neural message passing for graph retrieval implementation.

* Pau Riba, Andreas Fischer, Josep Lladós and Alicia Fornes, Learning Graph Distances with Message PassingNeural Networks, ICPR2018

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

Test a trained models:

#### Letters

* LOW

```
    $ python train_siamese_distance.py $DATA_PATH letters -t -l ./trained_models/LETTERS/LOW/feat_SoftHd_l3_h64.pth --distance SoftHd -b 128 --hidden_size 64 --nlayers 3 --representation feat
    Test distance:
        * 1-NN; Average Acc 97.867; Avg Time x Batch 0.480
        * 3-NN; Average Acc 98.000; Avg Time x Batch 0.480
        * 5-NN; Average Acc 98.267; Avg Time x Batch 0.480
```

* MED

```
    $ python train_siamese_distance.py $DATA_PATH letters -t -l ./trained_models/LETTERS/MED/feat_SoftHd_l3_h64.pth --distance SoftHd -b 128 --hidden_size 64 --nlayers 3 --representation feat
    Test distance:
        * 1-NN; Average Acc 88.533; Avg Time x Batch 0.536
        * 3-NN; Average Acc 88.000; Avg Time x Batch 0.536
        * 5-NN; Average Acc 89.200; Avg Time x Batch 0.536
```

* HIGH

```
    $ python train_siamese_distance.py $DATA_PATH letters -t -l ./trained_models/LETTERS/HIGH/feat_SoftHd_l3_h64.pth --distance SoftHd -b 128 --hidden_size 64 --nlayers 3 --representation feat
    Test distance:
        * 1-NN; Average Acc 79.200; Avg Time x Batch 0.529
        * 3-NN; Average Acc 82.533; Avg Time x Batch 0.529
        * 5-NN; Average Acc 82.533; Avg Time x Batch 0.529
```

#### Histograph

* Keypoints

```
    $ python train_siamese_distance.py $DATA_PATH histograph -t -l ./trained_models/HistoGraph/01_Keypoint/feat_SoftHd_l3_h64.pth --distance SoftHd -b 128 --hidden_size 64 --nlayers 3 --representation feat
    Test distance:
        * 1-NN; Average Acc 87.413; Avg Time x Batch 3.246
        * 3-NN; Average Acc 87.413; Avg Time x Batch 3.246
        * 5-NN; Average Acc 83.217; Avg Time x Batch 3.246
```

* Projection

```
    $ python train_siamese_distance.py $DATA_PATH histograph -t -l ./trained_models/HistoGraph/05_Progection/feat_SoftHd_l3_h64.pth --distance SoftHd -b 128 --hidden_size 64 --nlayers 3 --representation feat
    Test distance:
        * 1-NN; Average Acc 79.021; Avg Time x Batch 1.979
        * 3-NN; Average Acc 77.622; Avg Time x Batch 1.979
        * 5-NN; Average Acc 70.629; Avg Time x Batch 1.979
```

<p align="center"><img src="https://github.com/priba/siamese_ged/blob/master/readme_plots/pipeline.png" width="400"></p>

## Bibliography
- [1] Gilmer *et al.*, [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf), arXiv, 2017.

## Authors

* [Pau Riba](http://www.cvc.uab.es/people/priba/) ([@priba](https://github.com/priba))
