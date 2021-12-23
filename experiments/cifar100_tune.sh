# bash experiments/cifar100_tune.sh n

# process inputs
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}
SPLIT=10
DATASET=CIFAR100
N_CLASS=100

###############################################################
# save directory
DATE=Dec_2
OUTDIR=outputs/${DATE}/${DATASET}/${SPLIT}-task

# debuging flags
OVERWRITE=0
DEBUG=0
MAXTASK=3

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
P_ITERS=10000

##################
###### NOTES #####
##################
# first, tune
# second, try KD feature distillation with only updating linear layers (no blocks, as in notebook)
# third, try hierarchical KD feature distillation

# process inputs
if [ $DEBUG -eq 1 ] 
then   
    MAXTASK=3
    SCHEDULE="2"
    P_ITERS=10
fi
mkdir -p $OUTDIR

MODELNAME=resnet32

if [ $GPUID -eq 1 ] 
then

    for BETA_SRP in 0
    do
        for MU_SRB in 0
        do

            python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                --memory 0  --model_name $MODELNAME --model_type resnet_srb --mu $MU_SRB --beta $BETA_SRP --block_size $BLOCKSIZE_SRP --KD \
                --learner_type kd --learner_name LWF_FRB_DFC \
                --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours-playgroundb-sweep/beta-${BETA_SRP}_mu-${MU_SRB}

        done
    done

    for BETA_SRP in 0.1 1 2
    do
        for MU_SRB in 1 0.1 0.05 0.01 0.005
        do

            python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                --memory 0  --model_name $MODELNAME --model_type resnet_srb --mu $MU_SRB --beta $BETA_SRP --block_size $BLOCKSIZE_SRP --KD \
                --learner_type kd --learner_name LWF_FRB_DFC \
                --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours-playgroundb-sweep/beta-${BETA_SRP}_mu-${MU_SRB}

        done
    done
fi
if [ $GPUID -eq 2 ] 
then
    for BETA_SRP in 0.05 0.01 5
    do
        for MU_SRB in 1 0.1 0.05 0.01 0.005
        do

            python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                --memory 0  --model_name $MODELNAME --model_type resnet_srb --mu $MU_SRB --beta $BETA_SRP --block_size $BLOCKSIZE_SRP --KD \
                --learner_type kd --learner_name LWF_FRB_DFC \
                --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours-playgroundb-sweep/beta-${BETA_SRP}_mu-${MU_SRB}

        done
    done
fi
if [ $GPUID -eq 3 ] 
then
    for BETA_SRP in 0.005 0.001 10
    do
        for MU_SRB in 1 0.1 0.05 0.01 0.005
        do

            python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                --memory 0  --model_name $MODELNAME --model_type resnet_srb --mu $MU_SRB --beta $BETA_SRP --block_size $BLOCKSIZE_SRP --KD \
                --learner_type kd --learner_name LWF_FRB_DFC \
                --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours-playgroundb-sweep/beta-${BETA_SRP}_mu-${MU_SRB}

        done
    done
fi
# if [ $GPUID -eq 3 ] 
# then
#     BETA_SRP=0.05
#     MU_SRB=0.1
#     # ours - hierarchical
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#                 --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#                 --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#                 --memory 0  --model_name $MODELNAME --model_type resnet_srb --mu $MU_SRB --beta $BETA_SRP --block_size $BLOCKSIZE_SRP --KD \
#                 --learner_type kd --learner_name LWF_FRB_DFB \
#                 --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ours
    
# fi