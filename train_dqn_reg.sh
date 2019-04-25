#!/bin/bash
while read -r seed
do
	while read -r beta
   	do
		while read -r lambda
		do
			sbatch dqn_reg.sh $seed $beta $lambda
   		done < lambda.txt
   	done < beta.txt
done < seed.txt