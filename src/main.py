import torch
import numpy as np
import time
import arguments
import copy
import random
import os
import torch.multiprocessing as mp
import queue

from calibration import Calibration
from node import Client
from server import Server
from workers import *
from dataLoader.dataLoaders import getAllDataLoaders

if __name__ == "__main__":
    
    mp.set_start_method('spawn')
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        devices = [torch.device("cuda:{}".format(i)) for i in range(n_devices)]
        cuda = True
    else:
        n_devices = 1 
        devices = [torch.device('cpu')]
        cuda = False

    os.environ["OMP_NUM_THREADS"] = "1"

    args = arguments.parser()
    num_nodes = args.nodes
    num_round = args.round
    num_local_epoch = args.local_epoch
    
    print("> Setting:", args)

    nodeIDs = [i for i in range(num_nodes)]
    nodeindices, testLoader = getAllDataLoaders(nodeIDs, num_nodes, args)
    n_train_processes = n_devices * args.n_procs
    trainIDs = ["Train Worker : {}".format(i) for i in range(n_train_processes)]
    trainQ = mp.Queue()
    resultQ = mp.Queue()
    testQ = mp.Queue()
    preQ = queue.Queue()

    # create test process
    processes = []
    p = mp.Process(target=gpu_test_worker, args=(testLoader, testQ, devices[0], args))
    p.start()
    processes.append(p)
    time.sleep(0.1)

    # create train processes
    for i, trainID in enumerate(trainIDs):
        p = mp.Process(target=gpu_train_worker, args=(trainID, trainQ, resultQ, devices[i%n_devices], args))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    # create pseudo server
    server = Server(args)
    server.set_initial_model()

    # create pseudo clients
    nodes = []
    for i, nodeID in enumerate(nodeIDs):
        nodes.append(Client(nodeID, nodeindices[nodeID], args))
    
    lr = args.lr

    for roundIdx in range(args.round+1)[1:]: #1~100 round
        
        # if roundIdx % 5 == 0:
        print(f"Round {roundIdx}", end=', ')

        cur_time = time.time()
        lr *= args.lr_decay
       
        # Randomly selected clients
        n_trainees = int(num_nodes*args.fraction)
        trainees = [nodes[i] for i in np.random.choice(np.arange(num_nodes), n_trainees, replace=False)]
            
        count = 0

        for i, node in enumerate(trainees):       
            # download model
            model = server.get_model()
            count += 1
            trainQ.put({'type': 'train', 'node': copy.deepcopy(node), 'lr':lr, \
                'model': copy.deepcopy(model)})
            
        for _ in range(count):
            msg = resultQ.get()
            weight = msg['weight']
            node_id = msg['id']
            node = nodes[node_id]

            # upload weights to server
            server.update_node_info(weight)
            del msg

        # aggregate uploaded weights
        server.avg_parameters()

        # if roundIdx % 5 == 0:
        preQ.put({'round': roundIdx, 'model': copy.deepcopy(server.get_model())})
        print(f"Elapsed Time : {time.time()-cur_time:.1f}")

    for _ in range(n_train_processes):
        trainQ.put('kill')
        
    # Train finished
    time.sleep(5)

    


    # Test start
    for i in range(preQ.qsize()):
        testQ.put(preQ.get())

    testQ.put('kill')


    for p in processes:
        p.join()

    
    Calibration(args,devices[0])
    
    
