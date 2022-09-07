# srun --gres gpu:1 -p debug -c 6 -J "test" --pty bash
# sbatch submit_x.sh
# process inputs
DEFAULTMODE=0
MODE=${1:-$DEFAULTMODE}
GPUID=0
SPLIT=20
DATASET=TinyImageNet
N_CLASS=200

######################################################################
# TO-DO                                                              #
# LOOK AT LONG TASK SEQUENCES                                        #
######################################################################
# save directory
DATE=colla_round-d
OUTDIR_START=_outputs/${DATE}/${DATASET}/${SPLIT}-task

# debuging flags
OVERWRITE=0
DEBUG=0
MAXTASK=-1

# hard coded inputs
REPEAT=3
SCHEDULE="100 150 200 250"
vis_flag=1
BS=128
WD=0.0002
MOM=0.9

# optimizer
OPT="SGD"
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
mkdir -p $OUTDIR_START

for LR in 1e-2
do
    for MEMORY in 0
    do
        for MODELNAME in resnet18
        do
            OUTDIR_MODEL=${OUTDIR_START}/${MODELNAME}/mem-${MEMORY}/lr-${LR}
            if [ $MODE -eq 1 ] 
            then

                # UB
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $N_CLASS --other_split_size 0  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet \
                    --learner_type default --learner_name NormalNN \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR_MODEL}/UB 

                # EWC
                MU=10
                EPS=0
                OUTDIR=${OUTDIR_MODEL}/ewc-mc
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name EWC_MC --mu $MU --beta $BETA --eps $EPS \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR
                
                # L2
                MU=0
                EPS=-0.05
                OUTDIR=${OUTDIR_MODEL}/mc-l2
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name EWC_MC_L1L2  --mu $MU --beta $BETA --eps $EPS \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR  
            fi    
            if [ $MODE -eq 2 ] 
            then    
                # Lwf with feature space KD
                MU_KD=1
                OUTDIR=${OUTDIR_MODEL}/lwf-mc-featkd
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name LWF_MC_FEATKD --mu $MU_KD \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR  
            fi    
            if [ $MODE -eq 3 ] 
            then

                # base
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet \
                    --learner_type default --learner_name NormalNN \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR_MODEL}/base 

                # LWF
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name LWF_MC \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR_MODEL}/lwf-mc
                
                

                
     
            fi
        done
    done
done