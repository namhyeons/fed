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
writer = SummaryWriter('runs/cifar100_test')


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
       
        model1.load_state_dict(torch.load('model.pt'))
        
        model2.load_state_dict(torch.load('model.pt'))
        
        model3.load_state_dict(torch.load('model.pt'))

        model4.load_state_dict(torch.load('model.pt'))

        model5.load_state_dict(torch.load('model.pt'))

        # calibrate only 7th layer
        model1.train()
        model1.to(device)
        #freeze Layer 1~6
        model1.conv1.weight.requires_grad = False
        model1.conv2.weight.requires_grad = False
        model1.linear1.weight.requires_grad = False
        model1.linear2.weight.requires_grad = False
        model1.linear3.weight.requires_grad = False
        model1.linear4.weight.requires_grad = False
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
        model2.conv1.weight.requires_grad = False
        model2.conv2.weight.requires_grad = False
        model2.linear1.weight.requires_grad = False
        model2.linear2.weight.requires_grad = False
        model2.linear3.weight.requires_grad = False
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
        model3.conv1.weight.requires_grad = False
        model3.conv2.weight.requires_grad = False
        model3.linear1.weight.requires_grad = False
        model3.linear2.weight.requires_grad = False
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
        model1 = make_MobileNetV2(norm=args.norm)
        model1.load_state_dict(torch.load('model.pt'))
        model2 = make_MobileNetV2(norm=args.norm)
        model2.load_state_dict(torch.load('model.pt'))
        model3 = make_MobileNetV2(norm=args.norm)
        model3.load_state_dict(torch.load('model.pt'))
        model4 = make_MobileNetV2(norm=args.norm)
        model4.load_state_dict(torch.load('model.pt'))

        # calibrate only 7th layer
        model1.train()
        model1.to(device)
        #freeze all layer except classifier
        for g in model1.parameters():
            g.requires_grad = False
        model1.linear9.weight.requires_grad = True

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
        for g in model1.parameters():
            g.requires_grad = False
        model2.linear9.weight.requires_grad = True
        model2.conv8.weight.requires_grad = True
       
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







    







    




