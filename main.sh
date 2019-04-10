#!/bin/bash
while read -r seed
do
   while read -r est_beta
   do
      ./train.sh $seed $est_beta 
   done < beta.txt
done < seed.txt
