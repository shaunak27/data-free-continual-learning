#!/bin/sh -l
#SBATCH --gres gpu:1
#SBATCH --constraint=2080_ti 
#SBATCH -c 6
#SBATCH -p long
#SBATCH -t 150:00:00
#SBATCH -J dfcl_4
#SBATCH -o      _outputs/_log/dfcl_4.log
#SBATCH --error=_outputs/_log/dfcl_4.err

hostname
echo $CUDA_VISIBLE_DEVICES
srun bash experiments/cifar100-expand_long.sh 1