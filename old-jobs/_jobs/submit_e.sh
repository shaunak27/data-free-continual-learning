#!/bin/sh -l
#SBATCH --gres gpu:1
#SBATCH --constraint=2080_ti 
#SBATCH -p short
#SBATCH -c 6
#SBATCH -t 48:00:00
#SBATCH -J dfcl_5
#SBATCH -o      _log/dfcl_5.log
#SBATCH --error=_log/dfcl_5.err

hostname
echo $CUDA_VISIBLE_DEVICES
srun bash experiments/cifar100-continual.sh 5