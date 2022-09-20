#!/bin/bash

#SBATCH -A m3443_g -q regular
#SBATCH -C gpu 
#SBATCH -t 360:00
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -c 32
#SBATCH -o logs/%x-%j.out
#SBATCH -J GNN-train
#SBATCH --requeue
#SBATCH --gpu-bind=none
#SBATCH --signal=SIGUSR1@90

mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate rapids

export SLURM_CPU_BIND="cores"
echo -e "\nStarting sweeps\n"

# Single GPU training
srun -u python train_gnn.py $@