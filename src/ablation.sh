#!/bin/bash

python main.py --dataset='cifar10' --kd=0;
python main.py --dataset='cifar100' --kd=0;
python main.py --dataset='tiny-imagenet' --kd=0;
python main.py --dataset='cifar10' --iid=0;
python main.py --dataset='cifar100' --iid=0;
python main.py --dataset='tiny-imagenet' --iid=0;
python main.py --dataset='cifar10' --iid=0 --kd=0;
python main.py --dataset='cifar100' --iid=0 --kd=0;
python main.py --dataset='tiny-imagenet' --iid=0 --kd=0;