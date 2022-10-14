# bash experiments/domainnet.sh
# experiment settings
SPLIT=5
DATASET=DomainNet
N_CLASS=100

# save directory
DATE=sep_23
OUTDIR=_outputs/${DATE}/${DATASET}/${SPLIT}-task

# hard coded inputs
GPUID='0 1 2 3'
CONFIG_VIT=configs/domainnet_vit.yaml
CONFIG_VIT_P=configs/domainnet_vit_prompt.yaml
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

###########
#   NEW   #
###########

# l2p
python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name L2P \
    --prompt_param 18 6 1 \
    --log_dir ${OUTDIR}/vit/l2p_multi-layer

# dual prompt
python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name DualPrompt \
    --prompt_param 10 20 6 \
    --log_dir ${OUTDIR}/vit/dual-prompt

# #############
# # CONTINUAL #
# #############

# l2p
python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name L2P \
    --prompt_param 30 20 -1 \
    --log_dir ${OUTDIR}/vit/l2p

# linear only
python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name Linear \
    --log_dir ${OUTDIR}/vit/linear

# LWF
python -u run.py --config $CONFIG_VIT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type kd --learner_name LWF_MC \
    --log_dir ${OUTDIR}/vit/lwf-mc

# oracle
python -u run.py --config $CONFIG_VIT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type default --learner_name NormalNN --oracle_flag \
    --log_dir ${OUTDIR}/vit/oracle