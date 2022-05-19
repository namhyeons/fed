#!/bin/bash

python main.py --dataset='cifar100' --randomBranch=1;
python main.py --dataset='cifar100' --randomBranch=1 --kd=0;
python main.py --dataset='cifar100' --full=1;
python main.py --dataset='cifar100' --full=1 --kd=0;
python main.py --dataset='tiny-imagenet' --randomBranch=1;
python main.py --dataset='tiny-imagenet' --randomBranch=1 --kd=0;
python main.py --dataset='tiny-imagenet' --full=1;
python main.py --dataset='tiny-imagenet' --full=1 --kd=0;