# bash experiments/cifar100.sh
# experiment settings
SPLIT=10
DATASET=CIFAR100
N_CLASS=100

# save directory
DATE=sep_1b
OUTDIR=_outputs/${DATE}/${DATASET}/${SPLIT}-task

# hard coded inputs
GPUID='0 1 2 3'
CONFIG=configs/cifar100.yaml
CONFIG_VIT=configs/cifar100_vit.yaml
REPEAT=1
MEMORY=0
OVERWRITE=0
DEBUG=0

###############################################################

# process inputs
if [ $DEBUG -eq 1 ] 
then   
    MAXTASK=3
    SCHEDULE="2"
fi
mkdir -p $OUTDIR

# L2
python -u run.py --config $CONFIG_VIT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type kd --learner_name EWC_MC --eps 0.001 \
    --log_dir ${OUTDIR}/vit/l2-mc-sweep/1e-3

# L2
python -u run.py --config $CONFIG_VIT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type kd --learner_name EWC_MC --eps 0.01 \
    --log_dir ${OUTDIR}/vit/l2-mc-sweep/1e-2

# L2
python -u run.py --config $CONFIG_VIT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type kd --learner_name EWC_MC --eps 0.005 \
    --log_dir ${OUTDIR}/vit/l2-mc-sweep/5e-3

# L2
python -u run.py --config $CONFIG_VIT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type kd --learner_name EWC_MC --eps 0.0005 \
    --log_dir ${OUTDIR}/vit/l2-mc-sweep/5e-1

# L2
python -u run.py --config $CONFIG_VIT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type kd --learner_name EWC_MC --eps 0.05 \
    --log_dir ${OUTDIR}/vit/l2-mc-sweep/5e-2

# LWF
python -u run.py --config $CONFIG_VIT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type kd --learner_name LWF_MC \
    --log_dir ${OUTDIR}/vit/lwf-mc

# base
python -u run.py --config $CONFIG_VIT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type default --learner_name NormalNN \
    --log_dir ${OUTDIR}/vit/base 

# EWC
python -u run.py --config $CONFIG_VIT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type kd --learner_name EWC_MC --mu 10 \
    --log_dir ${OUTDIR}/vit/ewc-mc

# base
python -u run.py --config $CONFIG_VIT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type default --learner_name NormalNN --oracle_flag \
    --log_dir ${OUTDIR}/vit/oracle