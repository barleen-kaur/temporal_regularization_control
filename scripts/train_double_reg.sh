#!/bin/bash
IFS=","
while read -r env
do
	while read -r beta
	do
		while read -r lambda
		do
			sbatch double_reg.sh $env $beta $lambda
   		done < lambda.txt
	done < beta.txt
done < env.txt
