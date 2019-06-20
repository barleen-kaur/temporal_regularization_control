#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:25:00
#SBATCH --mem=8G
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=def-bengioy
#SBATCH --array=1-5

module load miniconda3
source activate temporal
cd ../temporal_regularization_control/

PYTHONPATH=.
echo "env=$1 seed=$SLURM_ARRAY_TASK_ID beta=$2 lambda=$3 freq=$4"
PYTHONPATH=${PYTHONPATH} python main.py --seed=$SLURM_ARRAY_TASK_ID --beta=$2 --lamb=$3  --env_name=$1 --algo dqn --plot_idx=$4 --disable_log True --no_cuda
