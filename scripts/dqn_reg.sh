#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=rpp-bengioy
#SBATCH --array=1-5%1 

module load miniconda3
source activate temporal_regularization_control
cd temporal_regularization_control/


PYTHONPATH=.
echo "algo=dqn env=$1 plot_freq=$2 seed=$SLURM_ARRAY_TASK_ID beta=$3 lambda=$4 lr=$5"
PYTHONPATH=${PYTHONPATH} python main.py --algo dqn --optim adam --lr=$5 --eps_decay 30000 --gamma 0.99 --seed=$SLURM_ARRAY_TASK_ID --replay_buff 100000 --plot_idx=$2 --target_idx 1000 --env_type atari --FA deep --env_name=$1 --num_frames 2000000 --frame_stack False --frame_skip 4 --replay_threshold 10000 --beta=$3 --lamb=$4 --return_deque 20 --disable_log True

