# bash experiments/cifar100_tune.sh n

# process inputs
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}
SPLIT=10
DATASET=CIFAR100
N_CLASS=100

###############################################################
# save directory
DATE=Jan_14
OUTDIR=outputs/${DATE}/${DATASET}/${SPLIT}-task

# debuging flags
OVERWRITE=0
DEBUG=0
MAXTASK=-1

# hard coded inputs
REPEAT=1
SCHEDULE="100 150 200 300"
vis_flag=1
BS=128
WD=0.0002
MOM=0.9

# optimizer
OPT="SGD"  
LR=0.1

# environment parameters
MEMORY=2000
MODELNAME=resnet32

#
# algorithm parameters
#
BLOCKSIZE_SRP=16
BETA_SRP=0
# SCHEDULE="1"
OVERWRITE=0
if [ $GPUID -eq 0 ] 
then
    MU_SRB=0.1
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME --model_type resnet_srb --mu $MU_SRB --beta $BETA_SRP --block_size $BLOCKSIZE_SRP --KD \
        --learner_type kd --learner_name LWF_FRB_DFC_lwfb \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours/a

    MU_SRB=0.05
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME --model_type resnet_srb --mu $MU_SRB --beta $BETA_SRP --block_size $BLOCKSIZE_SRP --KD \
        --learner_type kd --learner_name LWF_FRB_DFC_lwfb \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours/e

fi

if [ $GPUID -eq 1 ] 
then
    MU_SRB=0.01
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME --model_type resnet_srb --mu $MU_SRB --beta $BETA_SRP --block_size $BLOCKSIZE_SRP --KD \
        --learner_type kd --learner_name LWF_FRB_DFC_lwfb \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours/b
    MU=10
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name resnet32  --model_type resnet --KD --mu $MU \
        --learner_type kd --learner_name LWF_MC_ewc \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_ewc

fi

if [ $GPUID -eq 2 ] 
then
    MU_SRB=0.5
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME --model_type resnet_srb --mu $MU_SRB --beta $BETA_SRP --block_size $BLOCKSIZE_SRP --KD \
        --learner_type kd --learner_name LWF_FRB_DFC_lwfb \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours/c
    # LWF
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name resnet32  --model_type resnet --KD \
        --learner_type kd --learner_name LWF_MC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf
fi

if [ $GPUID -eq 3 ] 
then
    MU_SRB=1
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME --model_type resnet_srb --mu $MU_SRB --beta $BETA_SRP --block_size $BLOCKSIZE_SRP --KD \
        --learner_type kd --learner_name LWF_FRB_DFC_lwfb \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours/d
    MU=10
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name resnet32  --model_type resnet --KD --mu $MU \
        --learner_type kd --learner_name LWF_MC_ewc_b \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwfm_ewc

fi

# if [ $GPUID -eq 0 ] 
# then
#     MU=5
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name resnet32  --model_type resnet --KD --mu $MU \
#         --learner_type kd --learner_name LWF_MC_ewc \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_ewc/a
# fi

# if [ $GPUID -eq 1 ] 
# then
#     MU=10
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name resnet32  --model_type resnet --KD --mu $MU \
#         --learner_type kd --learner_name LWF_MC_ewc \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_ewc/b
# fi

# if [ $GPUID -eq 2 ] 
# then
#     MU=5
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name resnet32  --model_type resnet --KD --mu $MU \
#         --learner_type kd --learner_name LWF_MC_ewc \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_ewc/c
# fi

# if [ $GPUID -eq 3 ] 
# then
#     MU=10
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name resnet32  --model_type resnet --KD --mu $MU \
#         --learner_type kd --learner_name LWF_MC_ewc \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_ewc/d
# fi