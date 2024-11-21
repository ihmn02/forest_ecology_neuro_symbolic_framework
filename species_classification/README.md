# Species Classification
## Commands to Run
`srun -p gpu --gpus=a100:1 --mem=16gb --time=12:00:00  --pty -u bash -i`
`ml pytorch/2.0.1`
`. ~/set_comet_key.sh` 
`python3 test.py --wdir datasets/ns_rgb_fold0 --epochs 5 --batch 32 --tag r1r2 --aux_data chm --lr 1e-4 --thr 46.0 53.2 --scale 1.0 --seed 9874111 --lambdas 1.0 1.0`