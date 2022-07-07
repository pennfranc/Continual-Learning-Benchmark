import sys
import subprocess
import os

batch_size = 512
size_hidden = 20
reg_coef = 100
lrs = [1 / 10**(i / 2) for i in range(3, 13, 1)]
num_seeds = 1
log_name = f"lr-sweep-{batch_size=}-{size_hidden=}-{reg_coef=}"

if not os.path.exists("outputs/" + log_name):
    os.makedirs("outputs/" + log_name + "/bp")
    os.makedirs("outputs/" + log_name + "/ewc")

for lr in lrs:
    return_codes = []

    # ewc
    return_codes.append(subprocess.Popen(f"python3 -u iBatchLearn.py --gpuid -1 --repeat {num_seeds} --optimizer Adam  --schedule 4  --force_out_dim 2 --first_split_size 2 --other_split_size 2 --batch_size {batch_size} --model_name MLP{size_hidden} --agent_type customization  --agent_name EWC_mnist        --lr {lr} --reg_coef {reg_coef}    | tee outputs/{log_name}/ewc/{lr=}.log", shell=True))
    
    # bp
    return_codes.append(subprocess.Popen(f"python3 -u iBatchLearn.py --gpuid -1 --repeat {num_seeds} --optimizer Adam  --schedule 4  --force_out_dim 2 --first_split_size 2 --other_split_size 2 --batch_size {batch_size} --model_name MLP{size_hidden}        --lr {lr} --reg_coef 0    | tee outputs/{log_name}/bp/{lr=}.log", shell=True))

    exit_codes = [p.wait() for p in return_codes]