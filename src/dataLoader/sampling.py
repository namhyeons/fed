import numpy as np

def iid(dataset, num_nodes):
    np.random.seed(41)
    num_sample = int(len(dataset)/(num_nodes))
    dict_nodes = {}
    index = [i for i in range(len(dataset))]
    for i in range(num_nodes):
        dict_nodes[i] = np.random.choice(index, num_sample,
                                         replace=False)
        index = list(set(index)-set(dict_nodes[i]))
    return dict_nodes

def noniid(dataset, num_nodes, beta):
    labels = np.array([label for _, label in dataset])
    min_size = 0
    K = np.max(labels) + 1
    N = labels.shape[0]
    net_dataidx_map = {}
    n_nets = num_nodes
    np.random.seed(31)

    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_nets))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = np.array(idx_batch[j])
        #print(len(net_dataidx_map[j]))

    return net_dataidx_map

