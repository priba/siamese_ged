require "fileutils"

# path

path_data = "/home/ubelix/inf/andfischer/projects/pau2017/Letter/"
path_experiments = "/home/ubelix/inf/andfischer/projects/pau2017/experiments/"

# settings

dbs = ["LOW", "MED", "HIGH"]
layers = [2, 3]
edges = ["adj", "feat"]
distances = ["Hd", "SoftHd"]
hstates = [2, 64]

runs = 5
epochs = 100
batch_size = 128
lr = 1e-4
pipeline = "siamese_distance"

# params

params = []
dbs.each do |db|
  edges.each do |edge|
    distances.each do |distance|
      layers.each do |layer|
        hstates.each do |hstate|
          (0...runs).each do |run|
            params << {"db" => db, "edge" => edge, "distance" => distance, "layer" => layer, "hstate" => hstate, "run" => run}
          end
        end
      end
    end
  end
end

# main

sge_task_id = 0
sge_task_id = ARGV[0].to_i if ARGV.size > 0
if sge_task_id == 0
  
  cnt = 0
  params.each do |param|
    cnt += 1
    puts "#{cnt}: #{param.to_a.join(",")}"
  end
  puts "params: #{params.size}"
  
elsif sge_task_id > 0
  
  param = params[sge_task_id-1]
  db = param["db"]
  edge = param["edge"]
  distance = param["distance"]
  layer = param["layer"]
  hstate = param["hstate"]
  run = param["run"]
  
  dir_db = "#{path_data}#{db}/"
  
  dir_param = "#{pipeline}/#{db}/#{edge}_#{distance}_l#{layer}_h#{hstate}/r#{run}/"
  dir_run = "#{path_experiments}run/#{dir_param}"
  dir_checkpoint = "#{path_experiments}checkpoint/#{dir_param}"
  dir_log = "#{path_experiments}log/#{dir_param}"
  FileUtils.makedirs(dir_run)
  
  puts "train NMP: #{dir_run} .."
  cmd = "python train_siamese_distance.py #{dir_db} letters -s #{dir_checkpoint} --log #{dir_log} -lr #{lr} --nlayers #{layer} --hidden_size #{hstate} -e #{epochs} -b #{batch_size} --representation #{edge} --schedule #{epochs} --distance #{distance} > #{dir_run}run.txt"
  puts cmd
  puts ".. done."
end
