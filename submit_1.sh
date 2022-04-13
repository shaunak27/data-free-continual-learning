#!/bin/sh -l
#SBATCH --gres gpu:1
#SBATCH --constraint=2080_ti 
#SBATCH -c 6
#SBATCH -p long
#SBATCH -t 150:00:00
#SBATCH -J dfcl_1
#SBATCH -o      _log/dfcl_1.log
#SBATCH --error=_log/dfcl_1.err

hostname
echo $CUDA_VISIBLE_DEVICES
srun bash experiments/cifar100-continual.sh 1