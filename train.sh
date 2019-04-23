#!/bin/bash
PYTHONPATH=.
echo "seed=$1 beta=$2"
PYTHONPATH=${PYTHONPATH} python main.py --seed=$1 --beta=$2 --lr 0.001 --eps_decay 500 --replay_buff 1000 --plot_idx 200 --target_idx 100 --env_type gym --FA linear --env_name MountainCar-v0  --num_frames 200000  --lamb 0.2 
