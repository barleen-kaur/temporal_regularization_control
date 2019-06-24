#!/bin/bash
IFS=","
while read -r env freq
do
	while read -r beta
	do
		while read -r lambda
		do
			while read -r lr
            do
				sbatch dqn_reg.sh $env $freq $beta $lambda $lr
			done < lr.txt
		done < lambda.txt
	done < beta.txt
done < env.txt

