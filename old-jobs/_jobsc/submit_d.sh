#!/bin/sh -l
#SBATCH --gres gpu:1
#SBATCH --constraint=2080_ti 
#SBATCH -p short
#SBATCH -c 6
#SBATCH -t 48:00:00
#SBATCH -J dfcl_4c
#SBATCH -o      _log/dfcl_4c.log
#SBATCH --error=_log/dfcl_4c.err

hostname
echo $CUDA_VISIBLE_DEVICES
srun bash experiments/cifar100-continual_pt.sh 4