# bash experiments/cifar100.sh n

# process inputs
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}
SPLIT=10
DATASET=CIFAR100
N_CLASS=100

######################################################################
# TO-DO                                                              #
# LOOK AT LONG TASK SEQUENCES                                        #
######################################################################
# save directory
DATE=mar-1
OUTDIR=outputs/${DATE}/${DATASET}/${SPLIT}-task

# debuging flags
OVERWRITE=0
DEBUG=0
MAXTASK=5

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

###############################################################

# process inputs
if [ $DEBUG -eq 1 ] 
then   
    MAXTASK=3
    SCHEDULE="2"
fi
mkdir -p $OUTDIR

# for LR in 1e-3 1e-4 1e-2
for LR in 1e-3
do
    OUTDIR_LR=${OUTDIR}/lr-${LR}
    # for MEMORY in 0 2000
    for MEMORY in 0
    do
        OUTDIR_MEM=${OUTDIR_LR}/mem-${MEMORY}
        for MODELNAME in resnet18_pt resnet18
        do
            OUTDIR_MODEL=${OUTDIR_MEM}/${MODELNAME}
            if [ $GPUID -eq 0 ] 
            then
                # EWC with l2 start
                EPS=-0.1
                MU=5
                BETA=0
                OUTDIR_SWEEP=${OUTDIR_MODEL}/ewc-mc-l2start/mu-${MU}_beta-${BETA}
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS  \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name EWC_MC_L2START --mu $MU --beta $BETA --eps $EPS \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR_SWEEP
                # EWC with no logit distill
                EPS=0
                MU=5
                BETA=0
                OUTDIR_SWEEP=${OUTDIR_MODEL}/ewc-mc-nologitkd/mu-${MU}_beta-${BETA}
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name EWC_MC --mu $MU --beta $BETA --eps $EPS --playground_flag --temp -1 \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR_SWEEP
                OUTDIR_SWEEP=${OUTDIR_MODEL}/ewc-mc-local/mu-${MU}_beta-${BETA}
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name EWC_MC --mu $MU --beta $BETA --eps $EPS --playground_flag --temp -2 \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR_SWEEP
                OUTDIR_SWEEP=${OUTDIR_MODEL}/ewc/mu-${MU}_beta-${BETA}
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name EWC_MC --mu $MU --beta $BETA --eps $EPS --playground_flag --temp -3 \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR_SWEEP
                OUTDIR_SWEEP=${OUTDIR_MODEL}/ewc-lwf/mu-${MU}_beta-${BETA}
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name EWC_MC --mu $MU --beta $BETA --eps $EPS --playground_flag --temp -4 \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR_SWEEP
                # Lwf with feature space KD and no logit distill
                MU=5e1
                OUTDIR_SWEEP=${OUTDIR_MODEL}/lwf-mc-featkd-nologitkd/mu-${MU}
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name LWF_MC_FEATKD --mu $MU --playground_flag --temp -1 \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR_SWEEP
                OUTDIR_SWEEP=${OUTDIR_MODEL}/lwf-mc-featkd-local/mu-${MU}
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name LWF_MC_FEATKD --mu $MU --playground_flag --temp -2 \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR_SWEEP
                OUTDIR_SWEEP=${OUTDIR_MODEL}/featkd/mu-${MU}
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name LWF_MC_FEATKD --mu $MU --playground_flag --temp -3 \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR_SWEEP
                OUTDIR_SWEEP=${OUTDIR_MODEL}/lwf-featkd/mu-${MU}
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name LWF_MC_FEATKD --mu $MU --playground_flag --temp -4 \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR_SWEEP
                # 2nd
                #
                # implement white board metrics
                # analyze cka/lsa plots for memory
                #
                #
                #
                # 3rd
                #
                # save compute by removing reg in early parameters
                #
                #
                #
                # later
                #
                # ewc only do postiives or negateves to calculate loss for gradient
                # ewc/distill to protect first 2/3 of layers, ortho last third
            fi
            if [ $GPUID -eq 1 ] 
            then
                for MU in 1 5 10 20 50 100 0
                do
                    for BETA in 1e-2 5e-3 1e-3 5e-4 1e-4 0
                    do
                        # EWC
                        EPS=0
                        OUTDIR_SWEEP=${OUTDIR_MODEL}/ewc-mc/mu-${MU}_beta-${BETA}
                        python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                            --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                            --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                            --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                            --learner_type kd --learner_name EWC_MC --mu $MU --beta $BETA --eps $EPS \
                            --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR_SWEEP
                    done
                done
            fi
            if [ $GPUID -eq 2 ] 
            then
                for MU_KD in 1e1 5e1 1e2
                do 
                    # Lwf with feature space KD
                    OUTDIR_SWEEP=${OUTDIR_MODEL}/lwf-mc-featkd/mu-${MU_KD}
                    python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                        --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                        --learner_type kd --learner_name LWF_MC_FEATKD --mu $MU_KD \
                        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR_SWEEP
                done
                for EPS in 1e-4 5e-5 1e-5
                do
                    for MU in 1 5 10 20 50 100 0
                    do
                        for BETA in 1e-3 1e-4 0
                        do
                            # EWC-L1
                            OUTDIR_SWEEP=${OUTDIR_MODEL}/ewc-mc-l1/mu-${MU}_beta-${BETA}_eps-${EPS}
                            python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                                --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                                --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                                --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                                --learner_type kd --learner_name EWC_MC_L1L2 --mu $MU --beta $BETA --eps $EPS \
                                --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR_SWEEP
                        done
                    done
                done
            fi
            if [ $GPUID -eq 3 ] 
            then
                # LWF
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                    --learner_type kd --learner_name LWF_MC \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR_MODEL}/lwf-mc
                # base
                python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                    --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                    --memory $MEMORY  --model_name $MODELNAME  --model_type resnet \
                    --learner_type default --learner_name NormalNN \
                    --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR_MODEL}/base
                for EPS in -0.01 -0.001 -0.0001
                do
                    for MU in 1 5 10 20 50 100 0
                    do
                        for BETA in 1e-3 1e-4 0
                        do
                            # EWC-L2
                            OUTDIR_SWEEP=${OUTDIR_MODEL}/ewc-mc-l2/mu-${MU}_beta-${BETA}_eps-${EPS}
                            python -u run_ucl.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                                --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
                                --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                                --memory $MEMORY  --model_name $MODELNAME  --model_type resnet --KD \
                                --learner_type kd --learner_name EWC_MC_L1L2  --mu $MU --beta $BETA --eps $EPS \
                                --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir $OUTDIR_SWEEP
                        done
                    done
                done
            fi
        done
    done
done