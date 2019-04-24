#!/bin/bash
while read -r seed
do
	./train_noreg.sh $seed
done < seed.txt
