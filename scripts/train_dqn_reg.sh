#!/bin/bash
IFS=","
export num=0.1
while read -r env freq
do
	while read -r beta
	do
		while read -r lambda
		do
			while read -r lr
                        do
				if [ "$beta" == 0.0 -a `echo "$lambda>$num"|bc` -eq 1 ]; then
					continue
				else
					sbatch ~/temporal_regularization_control/scripts/dqn_reg.sh $env $freq $beta $lambda $lr
   				fi
			done < ~/temporal_regularization_control/scripts/lr.txt
		done < ~/temporal_regularization_control/scripts/lambda.txt
	done < ~/temporal_regularization_control/scripts/beta.txt
done < ~/temporal_regularization_control/scripts/env.txt
