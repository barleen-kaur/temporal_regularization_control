#!/bin/bash
PYTHONPATH=.
echo "seed=$1 beta=0 lambda=0.1"
PYTHONPATH=${PYTHONPATH} python main.py --seed=$1 --beta=0  --env_name CartPole-v0 
