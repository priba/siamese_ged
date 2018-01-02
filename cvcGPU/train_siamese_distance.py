import os, sys

# path

path_data = "/home/priba/Datasets/IAM/Letter/"
path_experiments = "/home/priba/nmp_ged/experiments_finetune/"

# settings

dbs = ["LOW", "MED", "HIGH"]
layers = ["3"]
edges = ["adj", "feat"]
distances = ["SoftHd"]
hstates = ["64"]

runs = 5
epochs = "200"
batch_size = "128"
lr = str(1e-4)
pipeline = "siamese_distance"

# params

params = []
for db in dbs:
    for edge in edges:
        for distance in distances:
            for layer in layers:
                for hstate in hstates:
                    for run in range(0, runs):
                        params.append({"db": db, "edge": edge, "distance": distance, "layer": layer, "hstate": hstate, "run": str(run)})

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
    distance = param["distance"]
    layer = param["layer"]
    hstate = param["hstate"]
    run = param["run"]

    dir_db = path_data + db + "/"

    dir_param = pipeline + "/" + db + "/" + edge + "_" + distance + "_l" + layer + "_h" + hstate + "/r" + run + "/"
    dir_run = path_experiments + "run/" + dir_param
    dir_checkpoint = path_experiments + "checkpoint/" + dir_param
    dir_log = path_experiments + "log/" + dir_param
    if not os.path.isdir(dir_run):
        os.makedirs(dir_run)

    print "train NMP: " + dir_run + " .."
    cmd = "python train_siamese_distance.py " + dir_db + " letters -s " + dir_checkpoint + " --log " + dir_log + " -lr " + lr + " --nlayers " + layer + " --hidden_size " + hstate + " -e " + epochs + " -b " + batch_size + " --representation " + edge + " --schedule " + epochs + " --distance " + distance + " > " + dir_run + "run.txt"
    print cmd
    os.system(cmd)
    print ".. done."
