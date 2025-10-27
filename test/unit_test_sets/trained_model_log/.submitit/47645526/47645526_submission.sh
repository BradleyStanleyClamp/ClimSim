#!/bin/bash

# Parameters
#SBATCH --account=orchid
#SBATCH --cpus-per-task=1
#SBATCH --error=/home/users/bradlesc/projects/ClimSim/logs/p2.1.1/5/testing/dress_rehearsal/2025-10-22-16-09-51/.submitit/%j/%j_0_log.err
#SBATCH --exclude=gpuhost004,gpuhost009,gpuhost012
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_general
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/home/users/bradlesc/projects/ClimSim/logs/p2.1.1/5/testing/dress_rehearsal/2025-10-22-16-09-51/.submitit/%j/%j_0_log.out
#SBATCH --partition=orchid
#SBATCH --qos=orchid
#SBATCH --signal=USR2@120
#SBATCH --time=1440
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/users/bradlesc/projects/ClimSim/logs/p2.1.1/5/testing/dress_rehearsal/2025-10-22-16-09-51/.submitit/%j/%j_%t_log.out --error /home/users/bradlesc/projects/ClimSim/logs/p2.1.1/5/testing/dress_rehearsal/2025-10-22-16-09-51/.submitit/%j/%j_%t_log.err /home/users/bradlesc/projects/ClimSim/.venv/bin/python3 -u -m submitit.core._submit /home/users/bradlesc/projects/ClimSim/logs/p2.1.1/5/testing/dress_rehearsal/2025-10-22-16-09-51/.submitit/%j
