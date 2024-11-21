#!/bin/bash
#SBATCH --time=50:00:00  #hr:mn:ss
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=  # Where to send mail	
#SBATCH --job-name=cdrule_bo        # Job name
#SBATCH --output=cdrule_bo_%j.out   # Standard output and error log

# Load modules or your own conda environment here
module load git
module load pytorch/2.0.1
sleep 5

#./rules_helper_framework.sh
# arg1: search type
# arg2: file with random seed values
# arg3: how many different seed values to run
./rules_helper_framework.sh b random_seeds.txt 15
