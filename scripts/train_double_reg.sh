#!/bin/bash
IFS=","
while read -r env freq frames
do
	while read -r beta lambda lr
	do
		sbatch double_reg.sh $env $freq $frames $beta $lambda $lr
	done < hyp.txt
done < env.txt
