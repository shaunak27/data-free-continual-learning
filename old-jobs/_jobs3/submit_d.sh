#!/bin/sh -l
#SBATCH --gres gpu:1
#SBATCH --constraint=2080_ti 
#SBATCH -p short
#SBATCH -c 6
#SBATCH -t 48:00:00
#SBATCH -J dfcl_4d
#SBATCH -o      _log/dfcl_4d.log
#SBATCH --error=_log/dfcl_4d.err

hostname
echo $CUDA_VISIBLE_DEVICES
srun bash experiments/cifar100-expand_long.sh 4