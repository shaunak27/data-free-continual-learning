# sh experiments/ICCV/tinyimnet-twentytask.sh n

# process inputs
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}
SPLIT=10

###############################################################
# save directory
DATE=ICCV-shuffled
OUTDIR=outputs/${DATE}/coreset-free-replay-twentytask/TinyImageNet
DATAROOT=/home/jsmith762/sscl/data

# debuging flags
OVERWRITE=0
DEBUG=0
MAXTASK=-1

# hard coded inputs
REPEAT=1
SCHEDULE="100 150 200 250"
MODELNAME=resnet32
GENMODELNAME="Autoencoder_tinyimnet"
vis_flag=0
BS=128
WD=0.0002
MOM=0.9

# optimizer
OPT="SGD"  
LR=0.1

# environment parameters
MEMORY=0

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
R_FEATURE_WEIGHT=5e1
DI_VAR_SCALE_GEN=1e-3
DI_CONTENT_WEIGHT=1
DI_DECAY=1e3
DI_LR_GEN=1e-3
DI_POWER_ITERS_GEN=10000
DI_REFRESH_ITERS=5

###############################################################

# process inputs
if [ $DEBUG -eq 1 ] 
then   
    SCHEDULE="8 10"
    DI_POWER_ITERS=5
    DI_POWER_ITERS_GEN=10
    DI_REFRESH_ITERS=50
    REPEAT=1
    OUTDIR=${OUTDIR}/_debug
    MAXTASK=3
fi
mkdir -p $OUTDIR
    
#########################
#      NEW STUFF       #
#########################
if [ $GPUID -eq 1 ] 
then
    # LWF
    python -u run_ucl.py --dataset TinyImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT\
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --mu $MU --memory 0  --model_name $MODELNAME  --model_type resnet --KD \
        --learner_type kd --learner_name LWF \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_0
    python -u run_ucl.py --dataset TinyImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT\
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --mu $MU --memory 0  --model_name $MODELNAME  --model_type resnet --KD \
        --learner_type kd --learner_name LWF_MC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_mc
    python -u run_ucl.py --dataset TinyImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT\
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --mu $MU --memory 2000  --model_name $MODELNAME  --model_type resnet --KD \
        --learner_type kd --learner_name LWF \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_2000

    # LWF - Synthetic
    python -u run_ucl.py --dataset TinyImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT\
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --mu $MUStar --memory $MEMORY --model_name $MODELNAME  --model_type resnet \
        --learner_type deep_inv_gen --learner_name DeepInversionLWF  \
        --gen_model_name $GENMODELNAME --gen_model_type generator \
        --playground_flag \
        --refresh_iters $DI_REFRESH_ITERS --beta $BETA --power_iters $DI_POWER_ITERS_GEN --deep_inv_params 0 $DI_LR_GEN $R_FEATURE_WEIGHT $DI_VAR_SCALE_GEN 0 $DI_CONTENT_WEIGHT $DI_DECAY $CONF_T 0 \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-synthetic
fi 
if [ $GPUID -eq 2 ] 
then
    # E2E
    python -u run_ucl.py --dataset TinyImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT\
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --mu $MU --memory 2000  --model_name $MODELNAME  --model_type resnet --KD \
        --learner_type kd --learner_name ETE --DW \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/e2e_2000

    # BIC
    python -u run_ucl.py --dataset TinyImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT\
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --mu $MU --memory 2000  --model_name $MODELNAME  --model_type resnet --KD \
        --learner_type kd --learner_name BIC --DW \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/bic_2000
    
    # base
    python -u run_ucl.py --dataset TinyImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT\
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME  --model_type resnet \
        --learner_type default --learner_name NormalNN \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/base_0
    python -u run_ucl.py --dataset TinyImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT\
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 2000  --model_name $MODELNAME  --model_type resnet \
        --learner_type default --learner_name NormalNN \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/base_2000
fi 
if [ $GPUID -eq 3 ] 
then
    # Full Method
    python -u run_ucl.py --dataset TinyImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT\
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --mu $MUStar --memory $MEMORY --model_name $MODELNAME  --model_type resnet \
        --learner_type deep_inv_gen --learner_name DeepInversionGenBN_agem_l2  \
        --gen_model_name $GENMODELNAME --gen_model_type generator \
        --playground_flag \
        --refresh_iters $DI_REFRESH_ITERS --beta $BETA --power_iters $DI_POWER_ITERS_GEN --deep_inv_params 0 $DI_LR_GEN $R_FEATURE_WEIGHT $DI_VAR_SCALE_GEN 0 $DI_CONTENT_WEIGHT $DI_DECAY $CONF_T 0 \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours

    # DeepInv
    python -u run_ucl.py --dataset TinyImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT\
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --mu $MU --memory $MEMORY --model_name $MODELNAME  --model_type resnet \
        --learner_type deep_inv_gen --learner_name DeepInversionGenBN  \
        --gen_model_name $GENMODELNAME --gen_model_type generator \
        --refresh_iters $DI_REFRESH_ITERS --power_iters $DI_POWER_ITERS_GEN --deep_inv_params 0 $DI_LR_GEN $R_FEATURE_WEIGHT $DI_VAR_SCALE_GEN 0 $DI_CONTENT_WEIGHT $DI_DECAY 0 0 \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/deepinv

    # oracle
    python -u run_ucl.py --dataset TinyImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT\
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0   --model_name $MODELNAME  --model_type resnet \
        --learner_type default --learner_name NormalNN --oracle_flag \
        --vis_flag $vis_flag --overwrite 0 --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/Oracle
fi