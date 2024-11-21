#!/bin/bash

search_type=$1
seed_file=$2
num_iter=$3

epochs=7
ntrials=64

date
echo "Running Bayesian optimization study..."
for (( idx=1; idx<=$num_iter; idx++ ))
do
   echo "Bash loop $idx"
   echo ""
   
   seed=`awk "FNR == $idx" ${seed_file}`

     #bayesian search
     echo "   python3 train.py --epochs $epochs --htune $search_type --seed $seed --n_trials $ntrials --scale 1.0"

     python3 train.py --epochs $epochs --htune $search_type --seed $seed --n_trials $ntrials --scale 1.0

     echo ""
done
date
