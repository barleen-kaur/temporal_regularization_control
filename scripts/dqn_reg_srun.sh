#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --account=rpp-bengioy 

module load miniconda3
source activate temporal_regularization_control
cd temporal_regularization_control/


PYTHONPATH=.
echo "algo=dqn env=$1 plot_freq=$2 seed=1 beta=$3 lambda=$4 lr=$5"
PYTHONPATH=${PYTHONPATH} srun --exclusive --cpu-bind=cores -c1 --mem=8G python main.py --algo dqn --optim adam --lr=$5 --eps_decay 30000 --gamma 0.99 --seed 1 --replay_buff 100000 --plot_idx=$2 --target_idx 1000 --env_type atari --FA deep --env_name=$1 --num_frames 2000000 --frame_stack False --frame_skip 4 --replay_threshold 10000 --beta=$3 --lamb=$4 --return_deque 20 --disable_log True &

echo "algo=dqn env=$1 plot_freq=$2 seed=2 beta=$3 lambda=$4 lr=$5"
PYTHONPATH=${PYTHONPATH} srun --exclusive --cpu-bind=cores -c1 --mem=8G python main.py --algo dqn --optim adam --lr=$5 --eps_decay 30000 --gamma 0.99 --seed 2 --replay_buff 100000 --plot_idx=$2 --target_idx 1000 --env_type atari --FA deep --env_name=$1 --num_frames 2000000 --frame_stack False --frame_skip 4 --replay_threshold 10000 --beta=$3 --lamb=$4 --return_deque 20 --disable_log True &

echo "algo=dqn env=$1 plot_freq=$2 seed=3 beta=$3 lambda=$4 lr=$5"
PYTHONPATH=${PYTHONPATH} srun --exclusive --cpu-bind=cores -c1 --mem=8G python main.py --algo dqn --optim adam --lr=$5 --eps_decay 30000 --gamma 0.99 --seed 3 --replay_buff 100000 --plot_idx=$2 --target_idx 1000 --env_type atari --FA deep --env_name=$1 --num_frames 2000000 --frame_stack False --frame_skip 4 --replay_threshold 10000 --beta=$3 --lamb=$4 --return_deque 20 --disable_log True &

wait
