#!/bin/sh -l
#SBATCH --gres gpu:1
#SBATCH --constraint=2080_ti 
#SBATCH -c 6
#SBATCH -p long
#SBATCH -t 150:00:00
#SBATCH -J dfcl_7
#SBATCH -o      _outputs/_log/dfcl_7.log
#SBATCH --error=_outputs/_log/dfcl_7.err

hostname
echo $CUDA_VISIBLE_DEVICES
srun bash experiments/tiny-continual_pt.sh 1
srun bash experiments/tiny-continual_pt.sh 2
srun bash experiments/tiny-continual_pt.sh 3