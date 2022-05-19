# from multiprocessing import Pool
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pickle
import copy
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import arguments
from dataLoader import dataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
from nn_models.resnet import multi_resnet18_kd

def evaluate(model, test_loader, args, device):

    model.to(device)
    model.eval()

    loss, total, correct_multi= 0.0, 0.0, 0.0
    accuracy_single_list = list()
    
    for i in range(args.num_branch):
        accuracy_single_list.append(0)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            output_list, _ = model(images)

            ensemble_output = torch.stack(output_list, dim=2)
            ensemble_output = torch.sum(ensemble_output, dim=2) / len(output_list)
            
            _, pred_labels_multi = torch.max(ensemble_output, 1)
            pred_labels_multi = pred_labels_multi.view(-1)
            correct_multi += torch.sum(torch.eq(pred_labels_multi, labels)).item()

            for i, single in enumerate(output_list):  
                _, pred_labels_single = torch.max(single, 1)
                pred_labels_single = pred_labels_single.view(-1)
                accuracy_single_list[i] += torch.sum(torch.eq(pred_labels_single, labels)).item()
                
            total += len(labels)

        accuracy_multi = correct_multi/total

        for i in range(len(accuracy_single_list)):
            accuracy_single_list[i] /= total
        
    model.to(torch.device('cpu'))
    
    return accuracy_multi, accuracy_single_list, loss


def create_model():
    if args.dataset == 'cifar100':
        model = multi_resnet18_kd(n_blocks=4, num_classes=100, norm='bn')
        return nn.DataParallel(model)


def get_dataset(args):
    

    if args.dataset == 'cifar100':
        apply_transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761)),
                                  ]
        )
        apply_transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761)),
                                  ]
        )
        dir = '~/FedSD/data/cifar100'
        train_dataset = datasets.CIFAR100(dir, train=True, download=True,
                                         transform=apply_transform_train)
        test_dataset = datasets.CIFAR100(dir, train=False, download=True,
                                        transform=apply_transform_test)

        return train_dataset, test_dataset


class KLLoss(nn.Module):
    def __init__(self, args):
        self.args = args
        super(KLLoss, self).__init__()

    def forward(self, pred, label):
        T=self.args.temperature
        
        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        with torch.no_grad():
            target = target_data.detach().clone()

        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss


class LSLoss(nn.Module):
    def __init__(self, args):
        self.args = args
        super(LSLoss, self).__init__()

    def forward(self, pred, label):
        T=self.args.temperature
        correct_prob = 0.99
        K = pred.size(1)
        teacher_soft = torch.ones_like(pred)
        teacher_soft = teacher_soft*(1-correct_prob)/(K-1)

        for i in range(pred.shape[0]):
            teacher_soft[i ,label[i]] = correct_prob

        predict = F.log_softmax(pred/T,dim=1)
    
        target_data = torch.pow(teacher_soft, 1/T)
        target_data = torch.nn.functional.normalize(target_data, p=1.0)
        #target_data = F.softmax(teacher_soft/T,dim=1)
        target_data = target_data+10**(-7)
        with torch.no_grad():
            target = target_data.detach().clone()

        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss


if __name__ == "__main__":

    import torch.nn as nn
    import queue

    """main"""
    args = arguments.parser()
    device = torch.device("cuda:0")

    print("> Setting:", args)

    # load data
    
    train_dataset, test_dataset = get_dataset(args)
    acc_list = list()

    net = create_model()
    net.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size = 256, shuffle = True, num_workers = 16)
    test_loader = DataLoader(test_dataset, batch_size = 256, shuffle = False, num_workers = 16)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    criterion_kl = KLLoss(args).cuda()
    criterion_ls = LSLoss(args).cuda()
    preq = queue.Queue()

    for roundIdx in range(args.round+1)[1:]:
        
        start_time = time.time()

        if roundIdx == 50:
            for g in optimizer.param_groups:
                g['lr'] = 0.1 * g['lr']

        args.consistency_rampup = int(args.round * 0.3)
        current = np.clip(roundIdx, 0.0, args.consistency_rampup)
        phase = 1.0 - current / args.consistency_rampup
        consistency_weight = float(np.exp(-5.0 * phase * phase))

        print("Current Round : {}".format(roundIdx), end=', ')
        net.train()
        net.to(device)

        for batch_idx, (images, labels) in enumerate(train_loader):
            
            #print("new batch")
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output_list, _ = net(images)

            loss = 0.

            for i, branch_output in enumerate(output_list):
                    loss += criterion(branch_output, labels) ## true label, branch output loss
                    
                    # if args.ls == 1:
                    #     loss += 0.5 * criterion_ls(branch_output, labels)

                    if args.kd == 1:
                        for j in range(len(output_list)):
                            if j == i:
                                continue
                            else:
                                loss += consistency_weight * criterion_kl(branch_output, output_list[j].detach()) / (len(output_list) - 1)


            #output = net(images)[0][0]
            #loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()

        print(f"Elapsed Time : {time.time()-start_time:.1f}")

        if roundIdx % 5 == 0:
            preq.put({'round': roundIdx, 'model': copy.deepcopy(net.to('cpu'))})
        
        # net.eval()
        # loss, total, correct = 0.0, 0.0, 0.0

        # with torch.no_grad():
        #     for batch_idx, (images, labels) in enumerate(test_loader):
        #         images, labels = images.to(device), labels.to(device)

        #         output = net(images)[0][-1]
        #         loss_batch = criterion(output, labels)
        #         loss += loss_batch.item()

        #         _, pred_labels = torch.max(output, 1)
        #         pred_labels = pred_labels.view(-1)
        #         correct += torch.sum(torch.eq(pred_labels, labels)).item()
        #         total += len(labels)

        #     accuracy = correct/total

        # print('Accuracy : {}'.format(accuracy))
        # acc_list.append(accuracy)

    preq.put('kill')
    global_acc = {'multi':list(), 'single':list()}

    while True:
        msg = preq.get()

        if msg == 'kill':
            break

        else:
            
            model = msg['model']
            
            round = msg['round']
            
            acc_multi, acc_single, loss = evaluate(model, test_loader, args, device)
            global_acc['multi'].append(acc_multi)
            global_acc['single'].append(acc_single)
        
            print("Round: {} / Multi Accuracy: {}".format(round, acc_multi))
            print("Round: {} / Single Accuracy: {}".format(round, acc_single[0]))


    file_name = '../../save/{}/K[{}]_own.pkl'.\
        format(args.dataset, args.kd)

    with open(file_name, 'wb') as f:
        pickle.dump([global_acc], f)

