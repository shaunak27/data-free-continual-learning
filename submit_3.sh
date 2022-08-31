#!/bin/sh -l
#SBATCH --gres gpu:1
#SBATCH --constraint=2080_ti 
#SBATCH -c 6
#SBATCH -p long
#SBATCH -t 150:00:00
#SBATCH -J dfcl_3
#SBATCH -o      _outputs/_log/dfcl_3.log
#SBATCH --error=_outputs/_log/dfcl_3.err

hostname
echo $CUDA_VISIBLE_DEVICES
srun bash experiments/cifar100-expand.sh 1