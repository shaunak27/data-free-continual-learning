# srun --gres gpu:1 -p debug -c 6 -J "test" --pty bash
# sbatch submit_x.sh
# process inputs
DEFAULTMODE=0
MODE=${1:-$DEFAULTMODE}
GPUID=0
SPLITA=10
SPLITB=10
DATASET=CIFAR100
N_CLASS=100

######################################################################
# TO-DO                                                              #
# LOOK AT LONG TASK SEQUENCES                                        #
######################################################################
# save directory
DATE=colla-debug
OUTDIR=_outputs/${DATE}/${DATASET}/95-5-task

# debuging flags
OVERWRITE=0
DEBUG=0
MAXTASK=-1

# hard coded inputs
REPEAT=1
SCHEDULE="50 70"
vis_flag=1
BS=256
WD=0.0002
MOM=0.9

# optimizer
OPT="Adam"
BETA=0

# environment parameters
MEMORY=2000

###############################################################

# process inputs
if [ $DEBUG -eq 1 ] 
then   
    MAXTASK=3
    SCHEDULE="2"
fi
mkdir -p $OUTDIR

for LR in 1e-3
do
    for MEMORY in 0
    do
        for MODELNAME in resnet18
        do
            OUTDIR_MODEL=${OUTDIR}/${MODELNAME}/mem-${MEMORY}/lr-${LR}
            
            if [ $MODE -eq 1 ] 
            then
                # LWF
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLITA --other_split_size $SPLITB  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name LWF_MC_B \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR_MODEL}/lwf-mc
                
            fi
            if [ $MODE -eq 2 ]
            then

                # base
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLITA --other_split_size $SPLITB  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet \
                    --learner_type default --learner_name NormalNN \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR_MODEL}/base 
        
            fi
        done
    done
done