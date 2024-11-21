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
     echo "   python3 train_pl.py --wdir datasets/ns_rgb_fold0 --epochs $epochs --batch 32 --tag baseline --aux_data chm --lr 1e-4 --seed $seed"

     python3 train_pl.py --wdir datasets/ns_rgb_fold0 --epochs $epochs --batch 32 --tag baseline --aux_data chm --lr 1e-4 --baseline --seed $seed

     echo ""
done
date
