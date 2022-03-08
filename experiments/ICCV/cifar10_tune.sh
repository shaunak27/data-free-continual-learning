# sh experiments/cifar10_tune.sh n

# process inputs
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}
SPLIT=5

###############################################################
# save directory
DATE=11-09-2021
OUTDIR=outputs/${DATE}/coreset-free-replay-twotask/cifar-10

# debuging flags
OVERWRITE=0
DEBUG=0
MAXTASK=2

# hard coded inputs
REPEAT=1
SCHEDULE="50 70"
vis_flag=1
BS=256
WD=0
MOM=0.9

# optimizer
OPT="Adam"  
LR=0.001

# environment parameters
MEMORY=1000

#
# algorithm parameters
#

###############################################################

# process inputs
if [ $DEBUG -eq 1 ] 
then   
    OUTDIR=${OUTDIR}/_debug
    MAXTASK=2
    SCHEDULE="2"
fi
mkdir -p $OUTDIR

# 18 or 32
#MODELNAME=resnet18
MODELNAME=resnet18_cc
OUTDIR=${OUTDIR}/${MODELNAME}


###############
###  TUNING ###
###############

# BLOCKSIZE_SRP=8
# LF_SRB=0
# MU_SRB=1
# BETA_SRP=0.1

# if [ $GPUID -eq 0 ] 
# then
#     # BLOCKSIZE_SRP=16
#     # LF_SRB=0
#     MU_SRB=0.05
# fi
# if [ $GPUID -eq 1 ] 
# then
#     # BLOCKSIZE_SRP=8
#     # LF_SRB=1
#     MU_SRB=0.01
# fi
# if [ $GPUID -eq 2 ] 
# then
#     # BLOCKSIZE_SRP=4
#     # LF_SRB=2
#     MU_SRB=0.005
# fi

# # Blocksize sweep
# OUTDIR=${OUTDIR}/SWEEP/MU
# # # UB
# # python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
# #     --first_split_size 10 --other_split_size 0  --schedule $SCHEDULE --schedule_type decay --batch_size $BS  \
# #     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
# #     --memory 0  --model_name $MODELNAME  --model_type resnet_srb --block_size $BLOCKSIZE_SRP \
# #     --learner_type default --learner_name NormalNN \
# #     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${BLOCKSIZE_SRP}
# # ours - covariance
# python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     --memory 0  --model_name $MODELNAME  --model_type resnet_srb --mu $MU_SRB --layer_freeze $LF_SRB --beta $BETA_SRP --block_size $BLOCKSIZE_SRP \
#     --learner_type kd --learner_name LWF_FRB_DF_COV --load_model_dir ${OUTDIR}/load \
#     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MU_SRB}

BLOCKSIZE_SRP=8
LF_SRB=0
MU_SRB=0.1
BETA_SRP=0.1

if [ $GPUID -eq 0 ] 
then
    # BLOCKSIZE_SRP=16
    # LF_SRB=0
    # MU_SRB=10
    BETA_SRP=1
fi
if [ $GPUID -eq 1 ] 
then
    # BLOCKSIZE_SRP=8
    # LF_SRB=1
    # MU_SRB=1
    BETA_SRP=0.1
fi
if [ $GPUID -eq 2 ] 
then
    # BLOCKSIZE_SRP=4
    # LF_SRB=2
    # MU_SRB=0.1
    BETA_SRP=0.01
fi

# Blocksize sweep
OUTDIR=${OUTDIR}/SWEEP/BETA
# ours - covariance
python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --memory 0  --model_name $MODELNAME  --model_type resnet_srb --mu $MU_SRB --layer_freeze $LF_SRB --beta $BETA_SRP --block_size $BLOCKSIZE_SRP \
    --learner_type kd --learner_name LWF_FRB_DF_COV \
    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${BETA_SRP}
###############
### /TUNING ###
###############







# # orth constraints
# # check on res block outputs (linear or softmax or tanh? scale and shift necessary?)
# if [ $GPUID -eq 0 ] 
# then

#     # ours - no matching
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name $MODELNAME  --model_type resnet_srb --mu $MU_SRB --layer_freeze $LF_SRB --beta $BETA_SRP --block_size $BLOCKSIZE_SRP \
#         --learner_type kd --learner_name LWF_FRB_DF \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours-none

#     # base w/ our arch
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name $MODELNAME  --model_type resnet_srb --block_size $BLOCKSIZE_SRP \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours-base

#     # upper bound
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size 10 --other_split_size 0  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name $MODELNAME  --model_type resnet \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/UB

# fi

# if [ $GPUID -eq 1 ] 
# then

#     # base
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name $MODELNAME  --model_type resnet \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/base_0

#     # LWF
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --mu 1 --memory 0  --model_name $MODELNAME  --model_type resnet --KD \
#         --learner_type kd --learner_name LWF_MC \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_mc_0

#     # LWF  w/ our arch
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name $MODELNAME  --model_type resnet_srb --KD --mu $MU_SRB --block_size $BLOCKSIZE_SRP \
#         --learner_type kd --learner_name LWF_MC \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours-lwf_mc_0

# fi

# if [ $GPUID -eq 2 ] 
# then

#     # ours - mmd
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name $MODELNAME  --model_type resnet_srb --mu $MU_SRB --layer_freeze $LF_SRB --beta $BETA_SRP --block_size $BLOCKSIZE_SRP \
#         --learner_type kd --learner_name LWF_FRB_DF_MMD \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours-mmd

# fi

# if [ $GPUID -eq 3 ] 
# then

#     # ours - covariance
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name $MODELNAME  --model_type resnet_srb --mu $MU_SRB --layer_freeze $LF_SRB --beta $BETA_SRP --block_size $BLOCKSIZE_SRP \
#         --learner_type kd --learner_name LWF_FRB_DF_COV \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours-cov

# fi