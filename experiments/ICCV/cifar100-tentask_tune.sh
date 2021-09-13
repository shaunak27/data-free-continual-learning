# bash experiments/ICCV/cifar100-tentask_tune.sh n

# process inputs
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}
SPLIT=10

###############################################################
# save directory
DATE=ICCV-shuffled
OUTDIR=outputs/${DATE}/coreset-free-replay-tentask/CIFAR100-tune-c

## debuging flags
OVERWRITE=0
DEBUG=0
MAXTASK=5

# hard coded inputs
REPEAT=1
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

if [ $GPUID -eq 0 ] 
then
    MUStar_ARRAY=(1 5e-1 2e-1)
fi
if [ $GPUID -eq 1 ] 
then
    MUStar_ARRAY=(1e-1 5e-2 2e-2)
fi
if [ $GPUID -eq 2 ] 
then
    MUStar_ARRAY=(1e-2 5e-3 2)
fi       
for MUStar in  "${MUStar_ARRAY[@]}"
do

    python -u run_ucl.py --dataset CIFAR100 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --mu $MUStar --memory $MEMORY --model_name $MODELNAME  --model_type resnet \
        --learner_type deep_inv_gen --learner_name DeepInversionGenBN_agem_l2  \
        --gen_model_name $GENMODELNAME --gen_model_type generator \
        --playground_flag \
        --refresh_iters $DI_REFRESH_ITERS --beta $BETA --power_iters $DI_POWER_ITERS_GEN --deep_inv_params 0 $DI_LR_GEN $R_FEATURE_WEIGHT $DI_VAR_SCALE_GEN 0 $DI_CONTENT_WEIGHT $DI_DECAY $CONF_T 0 \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours/mu-${MUStar} --load_model_dir ${OUTDIR}/ours

done


