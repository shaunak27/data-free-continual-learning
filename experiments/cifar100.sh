# sh experiments/cifar100.sh n

# process inputs
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}
SPLIT=10
DATASET=CIFAR100
N_CLASS=100

###############################################################
# save directory
DATE=CVPR_fixed-c
OUTDIR=outputs/${DATE}/${DATASET}/${SPLIT}-task

# debuging flags
OVERWRITE=0
DEBUG=0
MAXTASK=-1

# hard coded inputs
REPEAT=1
SCHEDULE="100 150 200 250"
vis_flag=1
BS=128
WD=0.0002
MOM=0.9

# optimizer
OPT="SGD"  
LR=0.1

# environment parameters
MEMORY=2000

#
# algorithm parameters
#
BLOCKSIZE_SRP=8
MU_SRB=1
BETA_SRP=1
P_ITERS=10000

###############################################################

# process inputs
if [ $DEBUG -eq 1 ] 
then   
    MAXTASK=3
    SCHEDULE="2"
    P_ITERS=10
fi
mkdir -p $OUTDIR

MODELNAME=resnet32

# ours - hierarchical
python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --memory 0  --model_name $MODELNAME --model_type resnet_srb --mu $MU_SRB --beta $BETA_SRP --block_size $BLOCKSIZE_SRP --KD \
    --learner_type kd --learner_name LWF_FRB_DFB \
    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours_b

# if [ $GPUID -eq 0 ] 
# then

#     # LWF
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name resnet32  --model_type resnet --KD \
#         --learner_type kd --learner_name LWF_MC \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_0

#     # upper bound
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $N_CLASS --other_split_size 0  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name $MODELNAME  --model_type resnet \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/UB

#     # base
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name $MODELNAME  --model_type resnet \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/base_0

    

#     # base
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name $MODELNAME  --model_type resnet \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/rehearsal_${MEMORY}

#     # LWF
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
#         --learner_type kd --learner_name LWF_MC \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_${MEMORY}

#     # BiC
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --mu 1.0 --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
#         --learner_type kd --learner_name BIC --DW \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/bic_${MEMORY}

#     # SS-IL
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --mu 1.0 --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
#         --learner_type kd --learner_name SSIL \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ssil_${MEMORY}

#     # ABD
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --mu 0.1 --memory 0 --model_name $MODELNAME  --model_type resnet \
#         --learner_type deep_inv_gen --learner_name DeepInversionGenBN_agem_l2  \
#         --gen_model_name Autoencoder_cifar --gen_model_type generator \
#         --playground_flag \
#         --refresh_iters 5 --beta 1 --power_iters $P_ITERS --deep_inv_params 0 1e-3 5e1 1e-3 0 1 1e3 1 0 \
#         --vis_flag 0 --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/abd

#     # ours - hierarchical
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name resnet32_h  --model_type resnet \
#         --learner_type kd --learner_name LWF_NOISE_H \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours_h

# fi

# # if [ $GPUID -eq 1 ] 
# # then

    

# # fi

# # if [ $GPUID -eq 2 ] 
# # then

   

# # fi

# if [ $GPUID -eq 3 ] 
# then

#     # ours - hierarchical
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay 0 --mu 0.1 \
#         --memory 0  --model_name resnet32_h  --model_type resnet \
#         --learner_type kd --learner_name LWF_NOISE_HN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours_h-n_a

#     # ours - hierarchical
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD --mu 0.05 \
#         --memory 0  --model_name resnet32_h  --model_type resnet \
#         --learner_type kd --learner_name LWF_NOISE_HN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours_h-n_b

#     # ours - hierarchical
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD --mu 0.01 \
#         --memory 0  --model_name resnet32_h  --model_type resnet \
#         --learner_type kd --learner_name LWF_NOISE_HN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours_h-n_c

# fi


# # # LWF
# # python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
# #     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
# #     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
# #     --memory 0  --model_name $MODELNAME  --model_type resnet_srb --KD --block_size $BLOCKSIZE_SRP \
# #     --learner_type kd --learner_name LWF_MC \
# #     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-srb_0

# # ours - covariance
# # python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
# #     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
# #     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
# #     --memory 0  --model_name $MODELNAME  --model_type resnet_srb --mu $MU_SRB --layer_freeze $LF_SRB --beta $BETA_SRP --block_size $BLOCKSIZE_SRP \
# #     --learner_type kd --learner_name LWF_FRB_DF_COV \
# #     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours