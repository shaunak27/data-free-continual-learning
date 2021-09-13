# bash experiments/ICCV/cifar100-tentask_ablate.sh n

# process inputs
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}
SPLIT=10

###############################################################
# save directory
DATE=ICCV-shuffled
OUTDIR=outputs/${DATE}/coreset-free-replay-tentask/CIFAR100-ablate

# debuging flags
OVERWRITE=0
DEBUG=0
MAXTASK=-1

# hard coded inputs
REPEAT=3
SCHEDULE="100 150 200 250"
MODELNAME=resnet32
GENMODELNAME="Autoencoder_cifar"
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

# fixed - ours
MU=1
R_FEATURE_WEIGHT=5e1
DI_VAR_SCALE_GEN=1e-3
DI_CONTENT_WEIGHT=1
DI_DECAY=1e3
DI_LR_GEN=1e-3
DI_POWER_ITERS_GEN=10000
DI_REFRESH_ITERS=5
MUStar=1e-1
BETA=1
CONF_T=1

###############################################################

# process inputs
if [ $DEBUG -eq 1 ] 
then   
    SCHEDULE="3 4 5"
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

if [ $GPUID -eq 3 ] 
then

    # Remove Gradient Projections
    python -u run_ucl.py --dataset CIFAR100 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --mu $MUStar --memory $MEMORY --model_name $MODELNAME  --model_type resnet \
        --learner_type deep_inv_gen --learner_name DeepInversionGenBN_agem_l2_ablate_a  \
        --gen_model_name $GENMODELNAME --gen_model_type generator \
        --playground_flag \
        --refresh_iters $DI_REFRESH_ITERS --beta $BETA --power_iters $DI_POWER_ITERS_GEN --deep_inv_params 0 $DI_LR_GEN $R_FEATURE_WEIGHT $DI_VAR_SCALE_GEN 0 $DI_CONTENT_WEIGHT $DI_DECAY $CONF_T 0 \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --load_model_dir ${OUTDIR} --log_dir ${OUTDIR}/ablate_class-balancing

    # No fine tune CE loss
    python -u run_ucl.py --dataset CIFAR100 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --mu $MUStar --memory $MEMORY --model_name $MODELNAME  --model_type resnet \
        --learner_type deep_inv_gen --learner_name DeepInversionGenBN_agem_l2_ablate_d  \
        --gen_model_name $GENMODELNAME --gen_model_type generator \
        --playground_flag \
        --refresh_iters $DI_REFRESH_ITERS --beta $BETA --power_iters $DI_POWER_ITERS_GEN --deep_inv_params 0 $DI_LR_GEN $R_FEATURE_WEIGHT $DI_VAR_SCALE_GEN 0 $DI_CONTENT_WEIGHT $DI_DECAY $CONF_T 0 \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --load_model_dir ${OUTDIR} --log_dir ${OUTDIR}/ablate_ft-ce

    # Standard CE loss only
    python -u run_ucl.py --dataset CIFAR100 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --mu $MUStar --memory $MEMORY --model_name $MODELNAME  --model_type resnet \
        --learner_type deep_inv_gen --learner_name DeepInversionGenBN_agem_l2_ablate_e  \
        --gen_model_name $GENMODELNAME --gen_model_type generator \
        --playground_flag \
        --refresh_iters $DI_REFRESH_ITERS --beta $BETA --power_iters $DI_POWER_ITERS_GEN --deep_inv_params 0 $DI_LR_GEN $R_FEATURE_WEIGHT $DI_VAR_SCALE_GEN 0 $DI_CONTENT_WEIGHT $DI_DECAY $CONF_T 0 \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --load_model_dir ${OUTDIR} --log_dir ${OUTDIR}/standard-ce

    # Logit Distillation
    python -u run_ucl.py --dataset CIFAR100 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --mu $MU --memory $MEMORY --model_name $MODELNAME  --model_type resnet \
        --learner_type deep_inv_gen --learner_name DeepInversionGenBN_agem_l2_ablate_h  \
        --gen_model_name $GENMODELNAME --gen_model_type generator \
        --playground_flag \
        --refresh_iters $DI_REFRESH_ITERS --beta $BETA --power_iters $DI_POWER_ITERS_GEN --deep_inv_params 0 $DI_LR_GEN $R_FEATURE_WEIGHT $DI_VAR_SCALE_GEN 0 $DI_CONTENT_WEIGHT $DI_DECAY $CONF_T 0 \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --load_model_dir ${OUTDIR} --log_dir ${OUTDIR}/logit-distillation

fi

if [ $GPUID -eq 1 ] 
then

    # Distill Real data only
    python -u run_ucl.py --dataset CIFAR100 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --mu $MUStar --memory $MEMORY --model_name $MODELNAME  --model_type resnet \
        --learner_type deep_inv_gen --learner_name DeepInversionGenBN_agem_l2_ablate_b  \
        --gen_model_name $GENMODELNAME --gen_model_type generator \
        --playground_flag \
        --refresh_iters $DI_REFRESH_ITERS --beta $BETA --power_iters $DI_POWER_ITERS_GEN --deep_inv_params 0 $DI_LR_GEN $R_FEATURE_WEIGHT $DI_VAR_SCALE_GEN 0 $DI_CONTENT_WEIGHT $DI_DECAY $CONF_T 0 \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --load_model_dir ${OUTDIR} --log_dir ${OUTDIR}/ablate_fake-data-kd

    

fi

if [ $GPUID -eq 0 ] 
then

    # Distill Fake data only
    python -u run_ucl.py --dataset CIFAR100 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --mu $MUStar --memory $MEMORY --model_name $MODELNAME  --model_type resnet \
        --learner_type deep_inv_gen --learner_name DeepInversionGenBN_agem_l2_ablate_c  \
        --gen_model_name $GENMODELNAME --gen_model_type generator \
        --playground_flag \
        --refresh_iters $DI_REFRESH_ITERS --beta $BETA --power_iters $DI_POWER_ITERS_GEN --deep_inv_params 0 $DI_LR_GEN $R_FEATURE_WEIGHT $DI_VAR_SCALE_GEN 0 $DI_CONTENT_WEIGHT $DI_DECAY $CONF_T 0 \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --load_model_dir ${OUTDIR} --log_dir ${OUTDIR}/ablate_real-data-kd

    # Feature distillation
    python -u run_ucl.py --dataset CIFAR100 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --mu $MUStar --memory $MEMORY --model_name $MODELNAME  --model_type resnet \
        --learner_type deep_inv_gen --learner_name DeepInversionGenBN_agem_l2_ablate_f  \
        --gen_model_name $GENMODELNAME --gen_model_type generator \
        --playground_flag \
        --refresh_iters $DI_REFRESH_ITERS --beta $BETA --power_iters $DI_POWER_ITERS_GEN --deep_inv_params 0 $DI_LR_GEN $R_FEATURE_WEIGHT $DI_VAR_SCALE_GEN 0 $DI_CONTENT_WEIGHT $DI_DECAY $CONF_T 0 \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --load_model_dir ${OUTDIR} --log_dir ${OUTDIR}/feature-distillation

    # Soft Distillation
    python -u run_ucl.py --dataset CIFAR100 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --mu $MU --memory $MEMORY --model_name $MODELNAME  --model_type resnet \
        --learner_type deep_inv_gen --learner_name DeepInversionGenBN_agem_l2_ablate_g  \
        --gen_model_name $GENMODELNAME --gen_model_type generator \
        --playground_flag \
        --refresh_iters $DI_REFRESH_ITERS --beta $BETA --power_iters $DI_POWER_ITERS_GEN --deep_inv_params 0 $DI_LR_GEN $R_FEATURE_WEIGHT $DI_VAR_SCALE_GEN 0 $DI_CONTENT_WEIGHT $DI_DECAY $CONF_T 0 \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --load_model_dir ${OUTDIR} --log_dir ${OUTDIR}/soft-distillation

fi