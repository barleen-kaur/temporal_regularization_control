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
				if [ "$beta" == 0.0 -a "$lambda" -gt 0.1 ]; then
					continue
				else
					sbatch dqn_reg.sh $env $freq $beta $lambda $lr
   				fi
			done < lr.txt
		done < lambda.txt
	done < beta.txt
done < env.txt
