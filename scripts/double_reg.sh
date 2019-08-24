#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=rpp-bengioy
#SBATCH --array=1-10%1 

module load miniconda3
source activate temporal
cd ../temporal_regularization_control/


PYTHONPATH=.
echo "algo=double env=$1 plot_freq=$2 num_frames=$3 seed=$SLURM_ARRAY_TASK_ID beta=$4 lambda=$5 lr=$6"
PYTHONPATH=${PYTHONPATH} python main.py --algo double --optim adam --lr=$6 --eps_decay 30000 --gamma 0.99 --seed=$SLURM_ARRAY_TASK_ID --replay_buff 100000 --plot_idx=$2 --target_idx 1000 --env_type atari --FA deep --env_name=$1 --num_frames=$3 --frame_stack False --frame_skip 4 --replay_threshold 10000 --beta=$4 --lamb=$5 --return_deque 20 --checkpoint_idx 1000000 --disable_log True --log_dir $SLURM_TMPDIR

cp -r $SLURM_TMPDIR/double_results/ $SCRATCH/temporal/
