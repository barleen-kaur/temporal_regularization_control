#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:2
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=rpp-bengioy
#SBATCH --array=1-5

module load miniconda3
source activate temporal
cd ../temporal_regularization_control/


PYTHONPATH=.
echo "env=$1 seed=$SLURM_ARRAY_TASK_ID beta=$2 lambda=$3"
PYTHONPATH=${PYTHONPATH} python main.py --algo double --optim adam --lr 1e-4 --eps_decay 30000 --gamma 0.99 --seed=$SLURM_ARRAY_TASK_ID --replay_buff 100000 --plot_idx 10000 --target_idx 1000 --env_type atari --FA deep --env_name=$1 --num_frames 2000000 --frame_stack False --frame_skip 4 --beta=$2 --lamb=$3 --return_deque 20 --disable_log True
