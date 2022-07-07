import sys
import subprocess
import os

batch_size = 512
size_hidden = 20
reg_coef = 100
lrs = [1 / 10**(i / 2) for i in range(3, 15, 5)]
num_seeds = 1
log_name = f"ewc-{batch_size=}-{size_hidden=}-{reg_coef=}"

if not os.path.exists("outputs/" + log_name):
    os.makedirs("outputs/" + log_name)

for lr in lrs:
    subprocess.call(f"python3 -u iBatchLearn.py --gpuid -1 --repeat {num_seeds} --optimizer Adam  --schedule 4  --force_out_dim 2 --first_split_size 2 --other_split_size 2 --batch_size {batch_size} --model_name MLP{size_hidden} --agent_type customization  --agent_name EWC_mnist        --lr {lr} --reg_coef {reg_coef}    | tee outputs/{log_name}/{lr=}.log", shell=True)
