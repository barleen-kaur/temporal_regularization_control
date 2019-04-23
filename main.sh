#!/bin/bash
while read -r seed
do
   while read -r beta
   do
      ./train.sh $seed $beta 
   done < beta.txt
done < seed.txt
