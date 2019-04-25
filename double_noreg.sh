#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=09:00:00
#SBATCH --gres=gpu:2
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=16GB
#SBATCH --account=rpp-bengioy

source ~/python3/bin/activate
cd ../temporal_regularization_control/

PYTHONPATH=.
echo "seed=$1 beta=0 lambda=0.1"
PYTHONPATH=${PYTHONPATH} python main.py --seed=$1 --beta=0 --lamb=0.1 --env_name MountainCar-v0 --algo double --plot_idx 2000 --disable_log True
