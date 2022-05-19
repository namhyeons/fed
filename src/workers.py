import copy
import pickle
import os
import torch

from models import evaluate
from dataLoader.dataset import get_dataset

def mv_average_10(lst):
    mv_lst = []
    for i in range(len(lst)-10):
        mv_lst.append(round(sum(lst[i:i+10]) / 10, 4))
    return mv_lst


def mv_average_20(lst):
    mv_lst = []
    for i in range(len(lst)-20):
        mv_lst.append(round(sum(lst[i:i+20]) / 20, 4))
    return mv_lst


def gpu_train_worker(trainID, trainQ, resultQ, device, args):

	train_dataset, _ = get_dataset(args)

	while True:
		msg = trainQ.get()

		if msg == 'kill':
			break

		elif msg['type'] == 'train':
			
			processing_node = msg['node']
			model = msg['model']
			model_weight = processing_node.train(device, msg['lr'], model, train_dataset)
			result = {'weight':copy.deepcopy(model_weight), 'id':processing_node.nodeID}
			resultQ.put(result)
			
		del processing_node
		del model
		del model_weight
		del msg




	#print("train end")

def gpu_test_worker(test_loader, testQ, device, args):


	while True:
		msg = testQ.get()

		if msg == 'kill':
			break

		else:
			model = msg['model']
			round = msg['round']
			
			loss,acc = evaluate(model, test_loader, args, device)
	
			print("Round: {} / Loss:{} / Accuracy: {}".format(round,loss,acc))

	#save model weight
	if args.dataset == 'cifar10':
		if args.beta == 0.5:
		 torch.save(model.state_dict(),"cifar10_model_0.5.pt")
		if args.beta == 0.1:
		 torch.save(model.state_dict(),"cifar10_model_0.1.pt")
		if args.beta == 0.05:
		 torch.save(model.state_dict(),"cifar10_model_0.05.pt")

	elif args.dataset == 'cifar100':
		if args.beta == 0.5:
		 torch.save(model.state_dict(),"cifar100_model_0.5.pt")
		if args.beta == 0.1:
		 torch.save(model.state_dict(),"cifar100_model_0.1.pt")
