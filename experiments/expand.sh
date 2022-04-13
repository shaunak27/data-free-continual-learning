# srun --gres gpu:1 -p debug -c 6 -J "test" --pty bash
# sbatch submit_x.sh
# process inputs
DEFAULTMODE=0
MODE=${1:-$DEFAULTMODE}
GPUID=0
SPLIT=10
DATASET=CIFAR100
N_CLASS=100

######################################################################
# TO-DO                                                              #
# LOOK AT LONG TASK SEQUENCES                                        #
######################################################################
# save directory
DATE=colla
OUTDIR=_outputs/${DATE}/${DATASET}/${SPLIT}-task

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

for LR in 1e-3 1e-4 1e-2
do
    OUTDIR_LR=${OUTDIR}/lr-${LR}
    for MEMORY in 0 2000
    do
        OUTDIR_MEM=${OUTDIR_LR}/mem-${MEMORY}
        for MODELNAME in resnet18 resnet18_pt
        do
            OUTDIR_MODEL=${OUTDIR_MEM}/${MODELNAME}
            if [ $MODE -eq 1 ] 
            then

                # base
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet \
                    --learner_type default --learner_name NormalNN \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR_MODEL}/base   
                
                # EWC with no logit distill
                EPS=0
                MU=10
                OUTDIR=${OUTDIR_MODEL}/ewc-mc-nologitkd
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name EWC_MC --mu $MU --beta $BETA --eps $EPS --playground_flag --temp -1 \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR
                OUTDIR=${OUTDIR_MODEL}/ewc-mc-local
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name EWC_MC --mu $MU --beta $BETA --eps $EPS --playground_flag --temp -2 \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR
                OUTDIR=${OUTDIR_MODEL}/ewc
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name EWC_MC --mu $MU --beta $BETA --eps $EPS --playground_flag --temp -3 \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR
                OUTDIR=${OUTDIR_MODEL}/ewc-lwf
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name EWC_MC --mu $MU --beta $BETA --eps $EPS --playground_flag --temp -4 \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR
                
                # Lwf with feature space KD and no logit distill
                MU=5e1
                OUTDIR=${OUTDIR_MODEL}/lwf-mc-featkd-nologitkd
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name LWF_MC_FEATKD --mu $MU --playground_flag --temp -1 \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR
                OUTDIR=${OUTDIR_MODEL}/lwf-mc-featkd-local
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name LWF_MC_FEATKD --mu $MU --playground_flag --temp -2 \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR
                OUTDIR=${OUTDIR_MODEL}/featkd
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name LWF_MC_FEATKD --mu $MU --playground_flag --temp -3 \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR
                OUTDIR=${OUTDIR_MODEL}/lwf-featkd
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name LWF_MC_FEATKD --mu $MU --playground_flag --temp -4 \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR

            fi
            if [ $MODE -eq 2 ] 
            then
                # LWF
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name LWF_MC \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR_MODEL}/lwf-mc
                
                
                # Lwf with feature space KD
                MU_KD=5e1
                OUTDIR=${OUTDIR_MODEL}/lwf-mc-featkd
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name LWF_MC_FEATKD --mu $MU_KD \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR 

                # EWC with l2 start
                EPS=-0.05
                MU=10
                OUTDIR=${OUTDIR_MODEL}/ewc-mc-l2start
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS  \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name EWC_MC_L2START --mu $MU --beta $BETA --eps $EPS \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR
                
            fi
            if [ $MODE -eq 3 ] 
            then

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

            fi
            if [ $MODE -eq 4 ] 
            then
 
                # Freeze
                MU=0
                EPS=-999999
                OUTDIR=${OUTDIR_MODEL}/mc-freeze
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name EWC_MC --mu $MU --beta $BETA --eps $EPS \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR

            fi
            if [ $MODE -eq 5 ] 
            then

                # EWC-L2
                MU=10
                EPS=-0.05
                OUTDIR=${OUTDIR_MODEL}/ewc-mc-l2
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name EWC_MC_L1L2  --mu $MU --beta $BETA --eps $EPS \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR
                # L2
                MU=0
                OUTDIR=${OUTDIR_MODEL}/mc-l2
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name EWC_MC_L1L2  --mu $MU --beta $BETA --eps $EPS \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR
            
            fi
        done
    done
done