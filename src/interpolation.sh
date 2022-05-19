#!/bin/bash

for i in {1..50}; do python main.py --dataset='cifar10' --interpolation=1; done
for i in {1..50}; do python main.py --dataset='cifar100' --interpolation=1; done