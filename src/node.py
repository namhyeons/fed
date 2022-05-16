import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader, Subset


class Client:
    def __init__(self, nodeID, node_indices, args):
        self.nodeID = nodeID
        self.__node_indices = node_indices
        self.__args = args

    
    def train(self, device, lr, model, train_dataset):
        
        train_loader = DataLoader(Subset(train_dataset, self.__node_indices), \
            batch_size=self.__args.batch_size, shuffle=True)
        model.train()
        model.to(device)
    
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum = 0.9,weight_decay=1e-5) ## changed 1e-3 -> 1e-5
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(self.__args.local_epoch):
            for _, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                output = model(images)
                loss = criterion(output,labels)
                loss.backward()
                optimizer.step()

        model.to(torch.device('cpu'))
        
        weight = model.state_dict()
         


        return copy.deepcopy(weight)