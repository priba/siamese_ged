import os, sys

# path
#path_data = "/media/priba/PPAP/NeuralMessagePassing/data/GWHistoGraphs/Data/Word_Graphs/01_Skew/"
#path_data = "/home/riba/Datasets/HistoGraphRetrieval/01_GT/01_GW/"

path_data = "/home/riba/Datasets/HistoGraphRetrieval/01_GT/02_PAR/"
path_experiments = "/home/riba/Results/experiments_retrieval/PAR/"


#path_data = "/media/priba/PPAP/NeuralMessagePassing/data/IAM/Letter/"
#path_experiments = "/media/priba/PPAP/SwitzerlandStay/nmp_ged/experiments_normalize_distance/"

# settings
#dataset = "letters"
dataset = "histographretrieval"
#dbs = ["cv1", "cv2", "cv3", "cv4"]
dbs = ["cv1"]
#dbs = ["01_Keypoint", "05_Projection"]
#dbs = ["LOW", "MED", "HIGH"]
layers = ["3"]
edges = ["feat"]
distances = ["SoftHd"]
hstates = ["64"]

runs = 4
epochs = "200"
batch_size = "20"
#lr = str(1e-2)
lr = str(1e-2)
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

    if len(sys.argv) == 2:
        cmd = "python train_siamese_distance.py " + dir_db + " " + dataset + " -s " + dir_checkpoint + " --log " + dir_log + " -lr " + lr + " --nlayers " + layer + " --hidden_size " + hstate + " -e " + epochs + " -b " + batch_size + " --representation " + edge + " --schedule 20 100 " + epochs + " --distance " + distance + " > " + dir_run + "run.txt"
    elif sys.argv[2]=='test':
        cmd = "python train_siamese_distance.py " + dir_db + " " + dataset + " -t -l " + dir_checkpoint + "checkpoint.pth --log " + dir_log + " -lr " + lr + " --nlayers " + layer + " --hidden_size " + hstate + " -e " + epochs + " -b " + batch_size + " --representation " + edge + " --schedule 20 100 " + epochs + " --distance " + distance + " > " + dir_run + "test.txt"
    elif sys.argv[2]=='write':
        dir_write = path_experiments + "data/" + dir_param
        dir_checkpoint = dir_checkpoint + 'checkpoint.pth'
        cmd = "python train_siamese_distance.py " + dir_db + " " + dataset + " -t -l " + dir_checkpoint + " -lr " + lr + " --nlayers " + layer + " --hidden_size " + hstate + " -e " + epochs + " -b " + batch_size + " --representation " + edge + " --schedule " + epochs + " --distance " + distance + " --write " + dir_write
    
    print cmd
    os.system(cmd)
    print ".. done."

