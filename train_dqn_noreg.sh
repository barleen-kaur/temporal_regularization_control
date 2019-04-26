#!/bin/bash
while read -r seed
do
	sbatch dqn_noreg.sh $seed
done < seed.txt