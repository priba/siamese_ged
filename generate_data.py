import os, sys

# path

path_data = "/media/priba/PPAP/NeuralMessagePassing/data/IAM/Letter/"
path_experiments = "/media/priba/PPAP/SwitzerlandStay/nmp_ged/AndreasResults/experiments/"

# settings

dbs = ["LOW", "MED", "HIGH"]
layers = ["3"]
edges = ["feat"]
hstates = ["64"]

runs = 5
epochs = "300"
batch_size = "128"
lr = str(1e-2)
pipeline = "learn_representation"
distance = "SoftHd"

# params

params = []
for db in dbs:
    for edge in edges:
        for layer in layers:
            for hstate in hstates:
                for run in range(0, runs):
                    params.append({"db": db, "edge": edge, "layer": layer, "hstate": hstate, "run": str(run)})

# main

if len(sys.argv) <= 1:
    cnt = 0
    for param in params:
        print cnt, ":", param
        cnt += 1
    print "params: ", cnt
else:
    sge_task_id = int(sys.argv[1])
    param = params[sge_task_id - 1]
    db = param["db"]
    edge = param["edge"]
    layer = param["layer"]
    hstate = param["hstate"]
    run = param["run"]

    dir_db = path_data + db + "/"

    dir_param = pipeline + "/" + db + "/" + edge + "_" + distance + "_l" + layer + "_h" + hstate + "/r" + run + "/"

    dir_data = path_experiments + "data/" + dir_param
    dir_checkpoint = path_experiments + "checkpoint/" + dir_param + 'checkpoint.pth'
    dir_log = path_experiments + "log/" + dir_param

    print "train NMP: " + dir_data + " .."
    cmd = "python train_learn_representation.py " + dir_db + " letters -t -l " + dir_checkpoint + " -lr " + lr + " --nlayers " + layer + " --hidden_size " + hstate + " -e " + epochs + " -b " + batch_size + " --representation " + edge + " --schedule 100 200 250 --distance " + distance + " --write " + dir_data
    print cmd
    os.system(cmd)
    print ".. done."
