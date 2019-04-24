#!/bin/bash
PYTHONPATH=.
echo "seed=$1 beta=$2 lambda=$3"
PYTHONPATH=${PYTHONPATH} python main.py --seed=$1 --beta=$2 --lamb $3 --env_name CartPole-v0 
