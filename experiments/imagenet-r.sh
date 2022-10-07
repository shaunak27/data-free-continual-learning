# bash experiments/imagenet-r.sh
# experiment settings
SPLIT=10
DATASET=ImageNet_R
N_CLASS=200

# save directory
DATE=oct_8
OUTDIR=_outputs/${DATE}/${DATASET}/${SPLIT}-task

# hard coded inputs
GPUID='0 1 2 3'
CONFIG_VIT=configs/imnet-r_vit.yaml
CONFIG_VIT_P_ATT=configs/imnet-r_vit_prompt_atte.yaml
CONFIG_VIT_P=configs/imnet-r_vit_prompt.yaml
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

# atteprompt
python -u run.py --config $CONFIG_VIT_P_ATT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name DualPrompt \
    --prompt_param 50 20 -1 3 \
    --log_dir ${OUTDIR}/vit/l2p_freeze-expand_ablate-att

# atteprompt
python -u run.py --config $CONFIG_VIT_P_ATT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name DualPrompt \
    --prompt_param 50 20 1 3 \
    --log_dir ${OUTDIR}/vit/l2p_freeze-expand

# atteprompt
python -u run.py --config $CONFIG_VIT_P_ATT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name DualPrompt \
    --prompt_param 25 10 1 3 \
    --log_dir ${OUTDIR}/vit/l2p_freeze-expand_small

# dual prompt
python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name DualPrompt \
    --prompt_param 10 20 6 -1 \
    --log_dir ${OUTDIR}/vit/dual-prompt














































# # l2p
# python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 55 20 1 3 \
#     --log_dir ${OUTDIR}/vit/l2p_freeze-expand_attention_ortho1

# # l2p
# python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 55 20 2 3 \
#     --log_dir ${OUTDIR}/vit/l2p_freeze-expand_attention_ortho2

# # l2p
# python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 55 20 7 3 \
#     --log_dir ${OUTDIR}/vit/l2p_freeze-expand_attention_ortho7

# # l2p
# python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 55 20 3 3 \
#     --log_dir ${OUTDIR}/vit/l2p_freeze-expand_attention_ortho3

# # l2p
# python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 55 20 4 3 \
#     --log_dir ${OUTDIR}/vit/l2p_freeze-expand_attention_ortho4

# # l2p
# python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 55 20 6 3 \
#     --log_dir ${OUTDIR}/vit/l2p_freeze-expand_attention_ortho6



















# # dual prompt
# python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 10 20 6 -1 \
#     --log_dir ${OUTDIR}/vit/dual-prompt

# # dual prompt
# python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 5 220 6 -1 \
#     --log_dir ${OUTDIR}/vit/dual-prompt-big


























# # dual prompt
# python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 10 20 6 2 \
#     --log_dir ${OUTDIR}/vit/dual-prompt_all-e

# # dual prompt
# python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 10 20 6 1 \
#     --log_dir ${OUTDIR}/vit/dual-prompt_no-g

# # # dual prompt
# # python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
# #     --learner_type prompt --learner_name DualPrompt \
# #     --prompt_param 10 20 6 -1 \
# #     --log_dir ${OUTDIR}/vit/dual-prompt

# # l2p
# python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name L2P \
#     --prompt_param 18 6 1 -1 \
#     --log_dir ${OUTDIR}/vit/l2p_multi-layer

# # l2p
# python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name L2P \
#     --prompt_param 30 20 -1 -1  \
#     --log_dir ${OUTDIR}/vit/l2p
    
# # # linear only
# # python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
# #     --learner_type prompt --learner_name Linear \
# #     --log_dir ${OUTDIR}/vit/linear

# # # LWF
# # python -u run.py --config $CONFIG_VIT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
# #     --learner_type kd --learner_name LWF_MC \
# #     --log_dir ${OUTDIR}/vit/lwf-mc

# # # oracle
# # python -u run.py --config $CONFIG_VIT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
# #     --learner_type default --learner_name NormalNN --oracle_flag \
# #     --log_dir ${OUTDIR}/vit/oracle