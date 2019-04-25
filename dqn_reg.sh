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
echo "seed=$1 beta=$2 lambda=$3"
PYTHONPATH=${PYTHONPATH} python main.py --seed=$1 --beta=$2 --lamb=$3  --env_name MountainCar-v0 --algo dqn --plot_idx 2000 --disable_log True
