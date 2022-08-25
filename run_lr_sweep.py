import subprocess
import os

batch_size = 512
size_hidden = 20
reg_coef_ewc = 100
reg_coef_si = 3000
lrs = [1 / 10**(i / 2) for i in range(3, 13, 1)]
num_seeds = 10
gpuid = 0
epochs = 2
log_name = f"lr-sweep-{batch_size=}-{size_hidden=}-{epochs=}"


for lr in lrs:
    return_codes = []

    # ewc
    # this is done like in scripts/split_MNIST_incremental_domain.sh but smaller net
    folder = "outputs/" + log_name + f"/ewc-{reg_coef_ewc=}"
    os.makedirs(folder, exist_ok=True)
    return_codes.append(subprocess.Popen(
        f"python3 -u iBatchLearn.py --gpuid {gpuid} --repeat {num_seeds} --optimizer Adam "
        f"--schedule {epochs}  --force_out_dim 2 --first_split_size 2 --other_split_size 2 "
        f"--batch_size {batch_size} --model_name MLP{size_hidden} --agent_type customization "
        f"--agent_name EWC_mnist --lr {lr} --reg_coef {reg_coef_ewc} "
        f"| tee {folder}/{lr=}.log",
        shell=True))
    
    # bp
    folder = "outputs/" + log_name + "/bp"
    os.makedirs(folder, exist_ok=True)
    return_codes.append(subprocess.Popen(
        f"python3 -u iBatchLearn.py --gpuid {gpuid} --repeat {num_seeds} --optimizer Adam "
        f"--schedule {epochs}  --force_out_dim 2 --first_split_size 2 --other_split_size 2 "
        f"--batch_size {batch_size} --model_name MLP{size_hidden} --lr {lr} --reg_coef 0 "
        f"| tee {folder}/{lr=}.log",
        shell=True))

    # si
    folder = "outputs/" + log_name + f"/si-{reg_coef_si=}"
    os.makedirs(folder, exist_ok=True)
    return_codes.append(subprocess.Popen(
        f"python3 -u iBatchLearn.py --gpuid {gpuid} --repeat {num_seeds} --optimizer Adam "
        f"--schedule {epochs}  --force_out_dim 2 --first_split_size 2 --other_split_size 2 "
        f"--batch_size {batch_size} --model_name MLP{size_hidden} --agent_type regularization "
        f"--agent_name SI --lr {lr} --reg_coef {reg_coef_si} "
        f"| tee {folder}/{lr=}.log",
        shell=True))

    exit_codes = [p.wait() for p in return_codes]
