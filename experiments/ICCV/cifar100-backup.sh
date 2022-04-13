# bash experiments/cifar100.sh n

# process inputs
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}
SPLIT=10
DATASET=CIFAR100
N_CLASS=100

######################################################################
# TO-DO                                                              #
# SWEEP LEARNING RATE (HIGHER,LOWER) AND LOOK AT LONG TASK SEQUENCES #
######################################################################
# save directory
DATE=feb-24
OUTDIR=outputs/${DATE}/${DATASET}/${SPLIT}-task

# debuging flags
OVERWRITE=0
DEBUG=0
MAXTASK=-1

# hard coded inputs
REPEAT=1
SCHEDULE="100 150 200 250"
vis_flag=1
BS=256
WD=0.0002
MOM=0.9

# optimizer
OPT="Adam"
LR=0.001

# environment parameters
MEMORY=2000

# model
MODELNAME=resnet18_pt

###############################################################

# process inputs
if [ $DEBUG -eq 1 ] 
then   
    MAXTASK=3
    SCHEDULE="2"
fi
mkdir -p $OUTDIR


if [ $GPUID -eq 0 ] 
then
    # LWF - ft-fancy
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME  --model_type resnet --KD \
        --learner_type kd --learner_name LWF_FT_MC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-ft-mc
fi
if [ $GPUID -eq 1 ] 
then
    # EWC-L1 - KD - MC
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME  --model_type resnet --KD --mu 1e-2 \
        --learner_type kd --learner_name EWCL1_KD_MC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ewl1-kd-mc



    # LWF - feat-dist
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME  --model_type resnet --KD \
        --learner_type kd --learner_name LWF_FD_MC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-fd-mc
    # EWC - KD - MC - L1 START _ L1 DISTANCE
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME  --model_type resnet --KD --mu 1e-3 \
        --learner_type kd --learner_name EWC_KD_MC_L1START_L1 \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ew-kd-mc-l1start_l1
fi

if [ $GPUID -eq 2 ] 
then

    # EWC-L1 - KD - MC
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name resnet18  --model_type resnet --KD --mu 1e-2 \
        --learner_type kd --learner_name EWCL1_KD_MC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ewl1-kd-mc-nopt



    # LWF
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME  --model_type resnet --KD \
        --learner_type kd --learner_name LWF_MC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-mc_0
    # EWC - KD - MC
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME  --model_type resnet --KD --mu 1 \
        --learner_type kd --learner_name EWC_KD_MC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ew-kd-mc
    # EWC - KD - MC
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name resnet18  --model_type resnet --KD --mu 1 \
        --learner_type kd --learner_name EWC_KD_MC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ew-kd-mc-nopt
    # EWC - KD - MC - L2 START
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME  --model_type resnet --KD --mu 1 \
        --learner_type kd --learner_name EWC_KD_MC_L2START \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ew-kd-mc-l2start

fi

if [ $GPUID -eq 3 ] 
then
    # L1 - KD - MC
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME  --model_type resnet --KD --mu 0.001 \
        --learner_type kd --learner_name L1_KD_MC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/l1-kd-mc


    # L2 - KD - MC
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name $MODELNAME  --model_type resnet --KD --mu 1 \
        --learner_type kd --learner_name L2_KD_MC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/l2-kd-mc
    # L2 - KD - MC
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name resnet18  --model_type resnet --KD --mu 1 \
        --learner_type kd --learner_name L2_KD_MC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/l2-kd-mc-nopt
    # LWF
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 0  --model_name resnet18  --model_type resnet --KD \
        --learner_type kd --learner_name LWF_MC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-mc-nopt_0
    # LWF
    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory 2000  --model_name $MODELNAME  --model_type resnet --KD \
        --learner_type kd --learner_name LWF_MC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-mc_2000
fi

# if [ $GPUID -eq 0 ] 
# then
#     # LWF
#     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name $MODELNAME  --model_type resnet --KD \
#         --learner_type kd --learner_name LWF_MC \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-mc_0

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

#     # FREEZE - KD - MC
#     for MU in 0 1 2 3
#     do
#         python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#             --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#             --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#             --memory 0  --model_name $MODELNAME  --model_type resnet --KD --mu $MU \
#             --learner_type kd --learner_name FREEZE_KD_MC \
#             --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/freeze-kd-mc/${MU}
#     done
    
#     # to-do: implement
#     # # FREEZE - EWC - KD - MC
#     # for MU in 0 1 2 3
#     # do
#     #     python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#     #         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#     #         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     #         --memory 0  --model_name $MODELNAME  --model_type resnet --KD --mu $MU \
#     #         --learner_type kd --learner_name FREEZE_EWC_KD_MC \
#     #         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/freeze-ewc-kd-mc/${MU}
#     # done
# fi

# if [ $GPUID -eq 1 ] 
# then
#     # L2 - KD - MC
#     for MU in 0.1 0.01 0.001 1 0.5 2 5 10
#     do
#         python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#             --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#             --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#             --memory 0  --model_name $MODELNAME  --model_type resnet --KD --mu $MU \
#             --learner_type kd --learner_name L2_KD_MC \
#             --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/l2-kd-mc/${MU}
#     done

#     # to-do: put EWC - MC back in
# fi

# if [ $GPUID -eq 2 ] 
# then
#     # EWC - KD - MC
#     for MU in 0.1 0.01 0.001 1 0.5 2 5 10
#     do
#         python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#             --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#             --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#             --memory 0  --model_name $MODELNAME  --model_type resnet --KD --mu $MU \
#             --learner_type kd --learner_name EWC_KD_MC \
#             --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ew-kd-mc/${MU}
#     done

#     # to-do: Smart pre-trained reg for first layers, ewc/lwf for last layer, mc
# fi

# if [ $GPUID -eq 3 ] 
# then
#     # EWC - KD - MC - L2 START
#     for MU in 0.1 0.01 0.001 1 0.5 2 5 10
#     do
#         python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#             --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#             --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#             --memory 0  --model_name $MODELNAME  --model_type resnet --KD --mu $MU \
#             --learner_type kd --learner_name EWC_KD_MC_L2START \
#             --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/ew-kd-mc-l2start/${MU}
#     done

#     # to-do: EWC - KD - MC - smart start
# fi