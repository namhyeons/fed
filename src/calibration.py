import torch
import torch.nn as nn
import torch.nn.functional as F
from dataLoader.dataset import get_dataset
from torch.utils.data import DataLoader
from models import evaluate
from nn_models.vggnet import make_VGG
from nn_models.vggnet import make_MobileNetV2
from model_avg import model_avg
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('0519/cifar10 beta=0.1')

def count_param(model):
    cnt = 0
    for name,param in model.named_parameters():
        if name in ['weight']:
            cnt += 1
    return cnt
            

def Calibration(args,device):

    train_dataset, test_dataset = get_dataset(args)

    #train
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    model1 = make_VGG(norm=args.norm)
    model2 = make_VGG(norm=args.norm)
    model3 = make_VGG(norm=args.norm)
    model4 = make_VGG(norm=args.norm)
    model5 = make_VGG(norm=args.norm)

    if (args.dataset == 'cifar10'):
        if args.beta == 0.5:
            model1.load_state_dict(torch.load('cifar10_model_0.5.pt'))
            model2.load_state_dict(torch.load('cifar10_model_0.5.pt'))
            model3.load_state_dict(torch.load('cifar10_model_0.5.pt'))
            model4.load_state_dict(torch.load('cifar10_model_0.5.pt'))
        elif args.beta == 0.1:
            model1.load_state_dict(torch.load('cifar10_model_0.1.pt'))
            model2.load_state_dict(torch.load('cifar10_model_0.1.pt'))
            model3.load_state_dict(torch.load('cifar10_model_0.1.pt'))
            model4.load_state_dict(torch.load('cifar10_model_0.1.pt'))
        elif args.beta == 0.05:
            model1.load_state_dict(torch.load('cifar10_model_0.05.pt'))
            model2.load_state_dict(torch.load('cifar10_model_0.05.pt'))
            model3.load_state_dict(torch.load('cifar10_model_0.05.pt'))
            model4.load_state_dict(torch.load('cifar10_model_0.05.pt'))
        

        # calibrate only classifier
        model1.train()
        model1.to(device)
        for name,param in model1.named_parameters():
            if 'linear5' in name:
                if 'weight' in name:
                    param.requires_grad = True
            else:
                param.requires_grad = False
        # print('111111111111111111111 :')
        # for g in model1.parameters():
        #     print(g)
            

        optimizer = torch.optim.SGD(filter(lambda f: f.requires_grad,model1.parameters()), lr=args.lr, momentum = 0.9,weight_decay=1e-5) ## changed 1e-3 -> 1e-5
        criterion = nn.CrossEntropyLoss()
        for index in range(100): # 100 epoch
            if (index == 70) :
                for g in optimizer.param_groups:
                    g['lr'] = 0.1 * g['lr']
            for _, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                model1.zero_grad()
                output = model1(images)
                loss = criterion(output,labels)
                loss.backward()
                optimizer.step()
            if((index+1)%5 == 0):
                model1.to(torch.device('cpu'))
                test_loader = DataLoader(test_dataset,256,shuffle=False)
                loss,acc = evaluate(model1, test_loader, args, device)
                writer.add_scalar("Loss/calibrate only 7th layer(classifier)",loss,index+1)
                writer.add_scalar("Accuracy/calibrate only 7th layer(classifier)",acc,index+1)
                print("Calibration only 7th layer // Loss:{} / Accuracy after calibration : {}".format(loss,acc))

        #calibrate 6,7th layer
        model2.train()
        model2.to(device)
        #freeze Layer 1~5
        for name,param in model2.named_parameters():
            if any(c in name for c in ('linear5','linear4')):
                if 'weight' in name:
                    param.requires_grad = True
            else:
                param.requires_grad = False
        # print('22222222222222222 :')
        # for g in model2.parameters():
        #     print(g)

        optimizer = torch.optim.SGD(filter(lambda f: f.requires_grad,model2.parameters()), lr=args.lr, momentum = 0.9,weight_decay=1e-5) ## changed 1e-3 -> 1e-5
        criterion = nn.CrossEntropyLoss()
        for index in range(100):  
            if (index == 70) :
                for g in optimizer.param_groups:
                    g['lr'] = 0.1 * g['lr']
            for _, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                model2.zero_grad()
                output = model2(images)
                loss = criterion(output,labels)
                loss.backward()
                optimizer.step()
            if((index+1)%5 == 0):
                model2.to(torch.device('cpu'))
                test_loader = DataLoader(test_dataset,256,shuffle=False)
                loss,acc = evaluate(model2, test_loader, args, device)
                writer.add_scalar("Loss/calibrate 6,7th layer",loss,index+1)
                writer.add_scalar("Accuracy/calibrate 6,7th layer",acc,index+1)
                print("Calibration 6,7th layer // Loss:{} / Accuracy after calibration : {}".format(loss,acc))



        

        #calibrate 5,6,7th layer
        model3.train()
        model3.to(device)
        #freeze Layer 1~4
        for name,param in model2.named_parameters():
            if any(c in name for c in ('linear5','linear4','linear3')):
                if 'weight' in name:
                    param.requires_grad = True
            else:
                param.requires_grad = False
        # print('333333333333333333333:')
        # for g in model2.parameters():
        #     print(g)
                
        optimizer = torch.optim.SGD(filter(lambda f: f.requires_grad,model3.parameters()), lr=args.lr, momentum = 0.9,weight_decay=1e-5) ## changed 1e-3 -> 1e-5
        criterion = nn.CrossEntropyLoss()
        for index in range(100):  
            if (index == 70) :
                for g in optimizer.param_groups:
                    g['lr'] = 0.1 * g['lr']
            for _, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                model3.zero_grad()
                output = model3(images)
                loss = criterion(output,labels)
                loss.backward()
                optimizer.step()
            if((index+1)%5 == 0):
                model3.to(torch.device('cpu'))
                test_loader = DataLoader(test_dataset,256,shuffle=False)
                loss,acc = evaluate(model3, test_loader, args, device)
                writer.add_scalar("Loss/calibrate 5,6,7th layer",loss,index+1)
                writer.add_scalar("Accuracy/calibrate 5,6,7th layer",acc,index+1)
                print("Calibration 5,6,7th layer // Loss:{} / Accuracy after calibration : {}".format(loss,acc))

        #calibrate all layer - supreme
        model4.train()
        model4.to(device)
        optimizer = torch.optim.SGD(model4.parameters(), lr=args.lr, momentum = 0.9,weight_decay=1e-5) ## changed 1e-3 -> 1e-5
        criterion = nn.CrossEntropyLoss()
        for index in range(100):  
            if (index == 70) :
                for g in optimizer.param_groups:
                    g['lr'] = 0.1 * g['lr']
            for _, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                model4.zero_grad()
                output = model4(images)
                loss = criterion(output,labels)
                loss.backward()
                optimizer.step()
            if((index+1)%5 == 0):
                model4.to(torch.device('cpu'))
                test_loader = DataLoader(test_dataset,256,shuffle=False)
                loss,acc = evaluate(model4, test_loader, args, device)
                writer.add_scalar("Loss/calibrate all layer",loss,index+1)
                writer.add_scalar("Accuracy/calibrate all layer",acc,index+1)
                print("Calibration all layer // Loss:{} / Accuracy after calibration : {}".format(loss,acc))
        writer.flush()
        writer.close()

    elif args.dataset == 'cifar100' :
        model1 = make_MobileNetV2()
        model2 = make_MobileNetV2()
        
        if args.beta == 0.5:
            model1.load_state_dict(torch.load('cifar100_model_0.5.pt'))
            model2.load_state_dict(torch.load('cifar100_model_0.5.pt'))
        elif args.beta == 0.1:
            model1.load_state_dict(torch.load('cifar100_model_0.1.pt'))
            model2.load_state_dict(torch.load('cifar100_model_0.1.pt'))
        elif args.beta == 0.05:
            model1.load_state_dict(torch.load('cifar100_model_0.05.pt'))
            model2.load_state_dict(torch.load('cifar100_model_0.05.pt'))
            



        # calibrate only 7th layer
        model1.train()
        model1.to(device)
        #freeze all layer except classifier
        cnt = count_param(model1)
        index = 0
        for name,param in model1.named_parameters() :
            if 'weight' in name:
                index += 1
                if index >= cnt:
                    param.requires_grad = True
            else:
                param.requires_grad = False
            
        optimizer = torch.optim.SGD(filter(lambda f: f.requires_grad,model1.parameters()), lr=args.lr, momentum = 0.9,weight_decay=1e-5) ## changed 1e-3 -> 1e-5
        criterion = nn.CrossEntropyLoss()
        for index in range(100): # 100 epoch
            if (index == 70) :
                for g in optimizer.param_groups:
                    g['lr'] = 0.1 * g['lr']
            for _, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                model1.zero_grad()
                output = model1(images)
                loss = criterion(output,labels)
                loss.backward()
                optimizer.step()
            if((index+1)%5 == 0):
                model1.to(torch.device('cpu'))
                test_loader = DataLoader(test_dataset,256,shuffle=False)
                loss,acc = evaluate(model1, test_loader, args, device)
                writer.add_scalar("Loss/calibrate only 7th layer(classifier)",loss,index+1)
                writer.add_scalar("Accuracy/calibrate only 7th layer(classifier)",acc,index+1)
                print("Calibration only 7th layer // Loss:{} / Accuracy after calibration : {}".format(loss,acc))
        
        #calibrate last 2 layers
        model2.train()
        model2.to(device)
        #freeze Layer except last 2 layers

        cnt = count_param(model2)
        index = 0
        for name,param in model1.named_parameters():
            if 'weight' in name:
                index += 1
                if index >= cnt - 1:
                    param.requires_grad = True
            else:
                param.requires_grad = False
            

        optimizer = torch.optim.SGD(filter(lambda f: f.requires_grad,model2.parameters()), lr=args.lr, momentum = 0.9,weight_decay=1e-5) ## changed 1e-3 -> 1e-5
        criterion = nn.CrossEntropyLoss()
        for index in range(100):  
            if (index == 70) :
                for g in optimizer.param_groups:
                    g['lr'] = 0.1 * g['lr']
            for _, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                model2.zero_grad()
                output = model2(images)
                loss = criterion(output,labels)
                loss.backward()
                optimizer.step()
            if((index+1)%5 == 0):
                model2.to(torch.device('cpu'))
                test_loader = DataLoader(test_dataset,256,shuffle=False)
                loss,acc = evaluate(model2, test_loader, args, device)
                writer.add_scalar("Loss/calibrate 6,7th layer",loss,index+1)
                writer.add_scalar("Accuracy/calibrate 6,7th layer",acc,index+1)
                print("Calibration 6,7th layer // Loss:{} / Accuracy after calibration : {}".format(loss,acc))







    







    



