# sh experiments/cifar10.sh n

# process inputs
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}
SPLIT=5

###############################################################
# save directory
DATE=11-02-2021b
OUTDIR=outputs/${DATE}/coreset-free-replay-twotask/cifar-10

# debuging flags
OVERWRITE=0
DEBUG=0
MAXTASK=2

# hard coded inputs
REPEAT=1
SCHEDULE="50 70"
vis_flag=1
BS=64
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
MODELNAME=resnet18
MODELNAME_CC=resnet18_cc

# if [ $GPUID -eq 0 ] 
# then

#     # multi-layer block internal feature replay - new architecture w/ mmd
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name $MODELNAME  --model_type resnet_srb \
#         --learner_type kd --learner_name LWF_FRB_DF_MMD \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/mlbrf_srp_data-free_mmd

# fi

# if [ $GPUID -eq 1 ] 
# then

#     # multi-layer block internal feature replay - new architecture w/ mmd
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name $MODELNAME_CC  --model_type resnet_srb \
#         --learner_type kd --learner_name LWF_FRB_DF_MMD \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/mlbrf_srp_data-free_mmd_cc

#     # multi-layer feature replay - new architecture
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name $MODELNAME  --model_type resnet_srb \
#         --learner_type kd --learner_name LWF_FRB_ABLATE \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/mlfr_srp_${MEMORY}

# fi

# if [ $GPUID -eq 2 ] 
# then

#     # Base with new architecture
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name $MODELNAME_CC  --model_type resnet_srb \
#         --learner_type kd --learner_name LWF_FRB_DF_MMD --mu 0 \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/mlbrf_srp_data-free_mmd_base_cc
    
#     # multi-layer block internal feature replay - new architecture w/ mmd & local ce
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name $MODELNAME  --model_type resnet_srb \
#         --learner_type kd --learner_name LWF_FRB_DF_MMD_L \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/mlbrf_srp_data-free_mmd_local-ce

# fi

# if [ $GPUID -eq 3 ] 
# then

#     # Base with new architecture
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name $MODELNAME  --model_type resnet_srb \
#         --learner_type kd --learner_name LWF_FRB_DF_MMD --mu 0 \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/mlbrf_srp_data-free_mmd_base

#     # feature replay
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name $MODELNAME  --model_type resnet \
#         --learner_type kd --learner_name LWF_FR \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/fr_${MEMORY}

#     # oracle
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0   --model_name $MODELNAME  --model_type resnet \
#         --learner_type default --learner_name NormalNN --oracle_flag \
#         --vis_flag $vis_flag --overwrite 0 --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/Oracle

#     # LWF
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --mu 1 --memory 0  --model_name $MODELNAME  --model_type resnet --KD \
#         --learner_type kd --learner_name LWF_MC \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/lwf_mc_0
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --mu 1 --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
#         --learner_type kd --learner_name LWF \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/lwf_${MEMORY}

#     # base
#     python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name $MODELNAME  --model_type resnet \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/base_${MEMORY}

# fi

if [ $GPUID -eq 0 ] 
then

    # multi-layer block internal feature replay - new architecture w/ mmd
    python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME  --model_type resnet_srb --mu 1 \
        --learner_type kd --learner_name LWF_FRB_DF_MMD_COV \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/mlbrf_srp_data-free_mmd_sweep/kd-1_cov

    # # multi-layer block internal feature replay - new architecture w/ mmd
    # python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
    #     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
    #     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    #     --memory 0  --model_name $MODELNAME  --model_type resnet_srb --mu 1 \
    #     --learner_type kd --learner_name LWF_FRB_DF_MMD \
    #     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/mlbrf_srp_data-free_mmd_sweep/kd-1

fi

if [ $GPUID -eq 1 ] 
then

    # multi-layer block internal feature replay - new architecture w/ mmd
    python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME  --model_type resnet_srb --mu 0.01 \
        --learner_type kd --learner_name LWF_FRB_DF_MMD \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/mlbrf_srp_data-free_mmd_sweep/kd-1e-2

fi

    
if [ $GPUID -eq 2 ] 
then

    # multi-layer block internal feature replay - new architecture w/ mmd
    python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME  --model_type resnet_srb --mu 0.05 \
        --learner_type kd --learner_name LWF_FRB_DF_MMD \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/mlbrf_srp_data-free_mmd_sweep/kd-5e-2

fi


if [ $GPUID -eq 3 ] 
then

    # # multi-layer block internal feature replay - new architecture w/ mmd
    # python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
    #     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
    #     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    #     --memory 0  --model_name $MODELNAME  --model_type resnet_srb --mu 1 \
    #     --learner_type kd --learner_name LWF_FRB_DF_MMD_VAR \
    #     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/mlbrf_srp_data-free_mmd_sweep/kd-1_var

    # multi-layer block internal feature replay - new architecture w/ mmd
    python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME_CC  --model_type resnet_srb --mu 1 \
        --learner_type kd --learner_name LWF_FRB_DF_MMD_PRO \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/mlbrf_srp_data-free_mmd_cc_sweep/kd-1_pro

    # # LWF
    # python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
    #     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
    #     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    #     --mu 1 --memory 0  --model_name $MODELNAME  --model_type resnet --KD \
    #     --learner_type kd --learner_name LWF_MC \
    #     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/lwf_mc_0
    # python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
    #     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
    #     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    #     --mu 1 --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
    #     --learner_type kd --learner_name LWF \
    #     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/lwf_${MEMORY}

    # # base
    # python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
    #     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
    #     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    #     --memory $MEMORY  --model_name $MODELNAME  --model_type resnet \
    #     --learner_type default --learner_name NormalNN \
    #     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/base_${MEMORY}

fi









# # multi-layer block internal feature replay - new architecture
# python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     --memory 0  --model_name $MODELNAME  --model_type resnet_srb \
#     --learner_type kd --learner_name LWF_FRB_DF \
#     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/mlbrf_srp_data-free

# # feature replay - new architecture
# python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     --memory $MEMORY  --model_name $MODELNAME  --model_type resnet_srb \
#     --learner_type kd --learner_name LWF_FR \
#     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/fr_srp_${MEMORY}

# # base - new architecture
# python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     --memory $MEMORY  --model_name $MODELNAME  --model_type resnet_srb \
#     --learner_type default --learner_name NormalNN \
#     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/base_srp_${MEMORY}

# # multi-layer block feature replay - new architecture
# python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     --memory $MEMORY  --model_name $MODELNAME  --model_type resnet_srb \
#     --learner_type kd --learner_name LWF_FRB \
#     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/mlbfr_srp_${MEMORY}

# # multi-layer block feature replay - new architecture - orthogonality contraints
# python -u run_ucl.py --dataset CIFAR10 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     --memory $MEMORY  --model_name $MODELNAME  --model_type resnet_srb \
#     --learner_type kd --learner_name LWF_FRB_OC \
#     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/${MODELNAME}/mlbrf_srp_oc_${MEMORY}