# sh experiments/mnist.sh n

# process inputs
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}
SPLIT=2

###############################################################
# save directory
DATE=09-21-2021
OUTDIR=outputs/${DATE}/coreset-free-replay-fivetask/mnist

# debuging flags
OVERWRITE=1
DEBUG=0
MAXTASK=-1

# hard coded inputs
REPEAT=1
SCHEDULE="10"
MODELNAME=MLP400
vis_flag=0
BS=128
WD=0
MOM=0.9

# optimizer
OPT="Adam"  
LR=0.001

# environment parameters
MEMORY=100

#
# algorithm parameters
#

###############################################################

# process inputs
if [ $DEBUG -eq 1 ] 
then   
    REPEAT=1
    # OUTDIR=${OUTDIR}/_debug
    MAXTASK=2
fi
mkdir -p $OUTDIR
    
# if [ $GPUID -eq 0 ] 
# then   

#     # feature replay - new architecture
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP200  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FR \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-fr_srp_${MEMORY}_model-sweep/200
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP300  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FR \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-fr_srp_${MEMORY}_model-sweep/300
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP400  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FR \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-fr_srp_${MEMORY}_model-sweep/400
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP500  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FR \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-fr_srp_${MEMORY}_model-sweep/500
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP100  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FR \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-fr_srp_${MEMORY}_model-sweep/100

#     # multi-layer feature replay - new architecture
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP200  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FRB_ABLATE \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-fr-ml_srp_${MEMORY}_model-sweep/200
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP300  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FRB_ABLATE \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-fr-ml_srp_${MEMORY}_model-sweep/300
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP400  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FRB_ABLATE \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-fr-ml_srp_${MEMORY}_model-sweep/400
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP500  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FRB_ABLATE \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-fr-ml_srp_${MEMORY}_model-sweep/500
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP100  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FRB_ABLATE \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-fr-ml_srp_${MEMORY}_model-sweep/100

# fi

# if [ $GPUID -eq 1 ] 
# then

    
#     # base - new architecture
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP200  --model_type mlp_srb \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/base_srp_${MEMORY}_model-sweep/200
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP300  --model_type mlp_srb \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/base_srp_${MEMORY}_model-sweep/300
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP400  --model_type mlp_srb \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/base_srp_${MEMORY}_model-sweep/400
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP500  --model_type mlp_srb \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/base_srp_${MEMORY}_model-sweep/500
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP100  --model_type mlp_srb \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/base_srp_${MEMORY}_model-sweep/100

#     # multi-layer block feature replay - new architecture
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP200  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FRB \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-frb-ml_srp_${MEMORY}_model-sweep/200
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP300  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FRB \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-frb-ml_srp_${MEMORY}_model-sweep/300
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP400  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FRB \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-frb-ml_srp_${MEMORY}_model-sweep/400
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP500  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FRB \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-frb-ml_srp_${MEMORY}_model-sweep/500
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP100  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FRB \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-frb-ml_srp_${MEMORY}_model-sweep/100

# fi

# if [ $GPUID -eq 2 ] 
# then

#     # base
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0  --model_name $MODELNAME  --model_type mlp \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/base_0
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP200  --model_type mlp \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/base_${MEMORY}_model-sweep/200
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP300  --model_type mlp \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/base_${MEMORY}_model-sweep/300
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP400  --model_type mlp \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/base_${MEMORY}_model-sweep/400
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP500  --model_type mlp \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/base_${MEMORY}_model-sweep/500
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP100  --model_type mlp \
#         --learner_type default --learner_name NormalNN \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/base_${MEMORY}_model-sweep/100

#     # feature replay
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP200  --model_type mlp \
#         --learner_type kd --learner_name LWF_FR \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-fr_${MEMORY}_model-sweep/200
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP300  --model_type mlp \
#         --learner_type kd --learner_name LWF_FR \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-fr_${MEMORY}_model-sweep/300
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP400  --model_type mlp \
#         --learner_type kd --learner_name LWF_FR \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-fr_${MEMORY}_model-sweep/400
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP500  --model_type mlp \
#         --learner_type kd --learner_name LWF_FR \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-fr_${MEMORY}_model-sweep/500
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP100  --model_type mlp \
#         --learner_type kd --learner_name LWF_FR \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-fr_${MEMORY}_model-sweep/100

# fi

# if [ $GPUID -eq 3 ] 
# then

#     # oracle
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory 0   --model_name $MODELNAME  --model_type mlp \
#         --learner_type default --learner_name NormalNN --oracle_flag \
#         --vis_flag $vis_flag --overwrite 0 --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/Oracle

#     # LWF
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --mu 1 --memory 0  --model_name $MODELNAME  --model_type mlp --KD \
#         --learner_type kd --learner_name LWF \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --mu 1 --memory 0  --model_name $MODELNAME  --model_type mlp --KD \
#         --learner_type kd --learner_name LWF_MC \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_mc_0
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --mu 1 --memory $MEMORY  --model_name $MODELNAME  --model_type mlp --KD \
#         --learner_type kd --learner_name LWF \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_${MEMORY}

#     # multi-layer block feature replay - new architecture - orthogonality contraints
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP200  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FRB_OC \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-frb-ml_srp_oc_${MEMORY}_model-sweep/200
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP300  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FRB_OC \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-frb-ml_srp_oc_${MEMORY}_model-sweep/300
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP400  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FRB_OC \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-frb-ml_srp_oc_${MEMORY}_model-sweep/400
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP500  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FRB_OC \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-frb-ml_srp_oc_${MEMORY}_model-sweep/500
#     python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --memory $MEMORY  --model_name MLP100  --model_type mlp_srb \
#         --learner_type kd --learner_name LWF_FRB_OC \
#         --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-frb-ml_srp_oc_${MEMORY}_model-sweep/100

# fi

if [ $GPUID -eq 0 ] 
then   

    # multi-layer block feature replay - new architecture - orthogonality contraints
    python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory $MEMORY  --model_name MLP400  --model_type mlp_srb \
        --learner_type kd --learner_name LWF_FRB_OC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-frb-ml_srp_oc_${MEMORY}_playground/a

fi

if [ $GPUID -eq 1 ] 
then

    
    # multi-layer block feature replay - new architecture - orthogonality contraints
    python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory $MEMORY  --model_name MLP400  --model_type mlp_srb \
        --learner_type kd --learner_name LWF_FRB_OC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-frb-ml_srp_oc_${MEMORY}_playground/b

fi

if [ $GPUID -eq 2 ] 
then

    # multi-layer block feature replay - new architecture - orthogonality contraints
    python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory $MEMORY  --model_name MLP400  --model_type mlp_srb \
        --learner_type kd --learner_name LWF_FRB_OC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-frb-ml_srp_oc_${MEMORY}_playground/c

fi

if [ $GPUID -eq 3 ] 
then

    # multi-layer block feature replay - new architecture - orthogonality contraints
    python -u run_ucl.py --dataset MNIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE --schedule_type decay --batch_size $BS    \
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --memory $MEMORY  --model_name MLP400  --model_type mlp_srb \
        --learner_type kd --learner_name LWF_FRB_OC \
        --vis_flag $vis_flag --overwrite $OVERWRITE --debug_mode $DEBUG --max_task $MAXTASK --log_dir ${OUTDIR}/lwf-frb-ml_srp_oc_${MEMORY}_playground/d

fi