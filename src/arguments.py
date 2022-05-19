import argparse


def parser():
    parser = argparse.ArgumentParser(description='Some hyperparameters')
    #federated arguments
    parser.add_argument('--nodes', type=int, default=10,
                        help='total number of nodes')
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='ratio of participating node')
    parser.add_argument('--round', type=int, default=100,
                        help='number of rounds')
    parser.add_argument('--local_epoch',  type=int, default=10,
                        help='number of local_epoch')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='size of batch')
    parser.add_argument('--norm', type=str, default='bn',
                        help='Default: Batch Normalization')                     
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.98,
                        help='0.992, 0.998')
    #other arguments
    parser.add_argument('--iid', type=int, default=0,
                        help='iid')
    parser.add_argument('--beta', type=float, default= 0.5,
                        help='beta for non iid dirichlet dist')                    
    parser.add_argument('--n_procs', type=int, default=5,
                        help='number of processes per GPU')
    parser.add_argument('--dataset',  type=str, default='cifar10',
                        help='type of dataset')
    # parser.add_argument('--base', type=int, default=-1,
    #                     help='Naive Baseline Experiments')
    # parser.add_argument('--ls', type=int, default=0,
    #                     help='whether using label smoothing regularization')
    # parser.add_argument('--ls_prob', type=float, default=0.99,
    #                     help='label smoothing correct probability')
    # parser.add_argument('--ls_alpha', type=float, default=0.1,
    #                     help='label   smoothing parameter')     
    # parser.add_argument('--sub_model', type=str, default='e',
    #                     help='HeteroFL experiment submodel')
                        
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser()
    print(args)
