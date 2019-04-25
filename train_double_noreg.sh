#!/bin/bash
while read -r seed
do
	sbatch double_noreg.sh $seed
done < seed.txt