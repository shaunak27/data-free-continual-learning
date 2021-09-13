# sh experiments/ICCV/imagenet-small-tentask.sh

# process inputs
DEFAULTGPU=0
GPUID="0 1"
SPLIT=10
UB_SPLIT=100

###############################################################
# save directory
DATE=ICCV-shuffled
OUTDIR=outputs/${DATE}/coreset-free-replay-tentask/ImageNet-50
DATAROOT=/datasets
# DATAROOT=/data/cwkuo/data

# debuging flags
OVERWRITE=0
DEBUG=0
MAXTASK=10

# hard coded inputs
REPEAT=1
SCHEDULE="30 60 80 90 100"
MODELNAME=resnet32
GENMODELNAME="Autoencoder_imnet"
vis_flag=0
WD=0.0001
MOM=0.9

# optimizer
OPT="SGD"  
# BS=256
# BS_GR=128
# LR=0.1
BS=128
BS_GR=64
LR=0.05

# environment parameters
MEMORY=0
MEMORY_REPLAY=2000

#
# algorithm parameters
#

# moving
MUStar=1e-1
BETA=1
CONF_T=1
SIG_SCALE=1 

# fixed - ours
MU=1
R_FEATURE_WEIGHT=1e2
DI_VAR_SCALE_GEN=1e-3
DI_CONTENT_WEIGHT=1
DI_DECAY=1e3
DI_LR_GEN=5e-4
DI_POWER_ITERS_GEN=50000
DI_REFRESH_ITERS=5

###############################################################

# process inputs
if [ $DEBUG -eq 1 ] 
then   
    SCHEDULE="1"
    DI_POWER_ITERS_GEN=10
    REPEAT=1
    OUTDIR=${OUTDIR}/_debug
    SPLIT=5
    MAXTASK=2
fi
mkdir -p $OUTDIR
    
#########################
#      NEW STUFF       #
#########################

# LWF
python -u run_ucl.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --mu $MU --memory 0  --model_name $MODELNAME  --model_type resnet --KD \
    --learner_type kd --learner_name LWF_MC \
    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_mc
python -u run_ucl.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --mu $MU --memory $MEMORY_REPLAY  --model_name $MODELNAME  --model_type resnet --KD \
    --learner_type kd --learner_name LWF \
    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_${MEMORY_REPLAY}
python -u run_ucl.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --mu $MU --memory 0  --model_name $MODELNAME  --model_type resnet --KD \
    --learner_type kd --learner_name LWF \
    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_0

# Full Method
python -u run_ucl.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS_GR    \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --mu $MUStar --memory $MEMORY --model_name $MODELNAME  --model_type resnet \
    --learner_type deep_inv_gen --learner_name DeepInversionGenBN_agem_l2  \
    --gen_model_name $GENMODELNAME --gen_model_type generator \
    --playground_flag \
    --refresh_iters $DI_REFRESH_ITERS --beta $BETA --power_iters $DI_POWER_ITERS_GEN --deep_inv_params 0 $DI_LR_GEN $R_FEATURE_WEIGHT $DI_VAR_SCALE_GEN 0 $DI_CONTENT_WEIGHT $DI_DECAY $CONF_T 0 \
    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours

# Upper Bound
python -u run_ucl.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
    --first_split_size $UB_SPLIT --other_split_size 0  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --memory 0   --model_name $MODELNAME  --model_type resnet \
    --learner_type default --learner_name NormalNN \
    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task 1 --log_dir ${OUTDIR}/UB

# base
python -u run_ucl.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --memory $MEMORY_REPLAY  --model_name $MODELNAME  --model_type resnet \
    --learner_type default --learner_name NormalNN \
    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/base_${MEMORY_REPLAY}

################
# LOW PRIORITY #
################

# # oracle
# python -u run_ucl.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
#     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     --memory 0   --model_name $MODELNAME  --model_type resnet \
#     --learner_type default --learner_name NormalNN --oracle_flag \
#     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/Oracle

# # DeepInv
# python -u run_ucl.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
#     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS_GR    \
#     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     --mu $MU --memory $MEMORY --model_name $MODELNAME  --model_type resnet \
#     --learner_type deep_inv_gen --learner_name DeepInversionGenBN  \
#     --gen_model_name $GENMODELNAME --gen_model_type generator \
#     --refresh_iters $DI_REFRESH_ITERS --power_iters $DI_POWER_ITERS_GEN --deep_inv_params 0 $DI_LR_GEN $R_FEATURE_WEIGHT $DI_VAR_SCALE_GEN 0 $DI_CONTENT_WEIGHT $DI_DECAY 0 0 \
#     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/deepinv

# # BIC
# python -u run_ucl.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
#     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     --mu $MU --memory $MEMORY_REPLAY  --model_name $MODELNAME  --model_type resnet --KD \
#     --learner_type kd --learner_name BIC --DW \
#     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/bic_${MEMORY_REPLAY}

# # E2E
# python -u run_ucl.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
#     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     --mu $MU --memory $MEMORY_REPLAY  --model_name $MODELNAME  --model_type resnet --KD \
#     --learner_type kd --learner_name ETE --DW \
#     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/e2e_${MEMORY_REPLAY}

# # base
# python -u run_ucl.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
#     --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     --memory 0  --model_name $MODELNAME  --model_type resnet \
#     --learner_type default --learner_name NormalNN \
#     --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/base_0

