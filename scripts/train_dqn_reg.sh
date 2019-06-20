#!/bin/bash
IFS=","
while read -r env freq
do
	while read -r beta
	do
		while read -r lambda
		do
			sbatch dqn_reg.sh $env $beta $lambda $freq
   		done < lambda.txt
	done < beta.txt
done < env.txt
