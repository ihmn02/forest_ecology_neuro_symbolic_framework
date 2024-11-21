#!/bin/bash

search_type=$1
seed_file=$2
num_iter=$3

epochs=5
ntrials=64

date
echo "Running Bayesian optimization study..."
for (( idx=1; idx<=$num_iter; idx++ ))
do
   echo "Bash loop $idx"
   echo ""
   
   seed=`awk "FNR == $idx" ${seed_file}`
     echo "ns_rgb_fold0"

     #bayesian search
     echo "   python3 train_pl.py --wdir datasets/ns_rgb_fold0 --epochs $epochs --batch 32 --tag r1r2 --aux_data chm --lr 1e-4 --thr 46.0 53.2 --scale 1.0 --htune $search_type --seed $seed --n_trials $ntrials"

     python3 train_pl.py --wdir datasets/ns_rgb_fold0 --epochs $epochs --batch 32 --tag r1r2 --aux_data chm --lr 1e-4 --thr 46.0 53.2 --scale 1.0 --htune $search_type --seed $seed --n_trials $ntrials

     echo ""
done
date
