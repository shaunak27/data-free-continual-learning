# bash experiments/imagenet-r.sh
# experiment settings
SPLIT=10
DATASET=ImageNet_R
N_CLASS=200

# save directory

# if [ $# -ge 2 ]
#     then
#         DATE="jan_21_newclipl2p_$2"
if [ $# -ge 1 ]
    then
        DATE="jan_21_newclipl2p_$1"
else
    DATE="jan_21_newclipl2p"
fi

OUTDIR=_outputs/${DATE}/${DATASET}/${SPLIT}-task

# hard coded inputs
GPUID='0 1 2 3'
CONFIG_VIT=configs/imnet-r_vit.yaml
CONFIG_VIT_P_ATT=configs/imnet-r_vit_prompt_atte.yaml
CONFIG_VIT_P=configs/imnet-r_vit_prompt.yaml
CONFIG_CLIP=configs/imnet-r_clip_prompt.yaml
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


# # atteprompt
# python -u run.py --config $CONFIG_VIT_P_ATT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 100 20 1 1 --mu 0 \
#     --log_dir ${OUTDIR}/vit/atteprompt_mu-0

# # atteprompt
# python -u run.py --config $CONFIG_VIT_P_ATT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 100 20 1 1 --mu 1 \
#     --log_dir ${OUTDIR}/vit/atteprompt_mu-1

# # atteprompt
# python -u run.py --config $CONFIG_VIT_P_ATT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 100 20 1 1 --mu 10 \
#     --log_dir ${OUTDIR}/vit/atteprompt_mu-10

# # atteprompt
# python -u run.py --config $CONFIG_VIT_P_ATT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 100 20 1 1 --mu 0.1 \
#     --log_dir ${OUTDIR}/vit/atteprompt_mu-0.1



MU=1
# ablate attention
# python -u run.py --config $CONFIG_VIT_P_ATT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 100 20 -1 1 --mu $MU \
#     --log_dir ${OUTDIR}/vit/atteprompt_ablate-att
# # small
# python -u run.py --config $CONFIG_VIT_P_ATT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 40 15 1 -1 --mu $MU \
#     --log_dir ${OUTDIR}/vit/atteprompt_small

# l2p
# python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name L2P \
#     --prompt_param 100 20 1 -1 \
#     --log_dir ${OUTDIR}/vit/l2p_multi-layer

# l2p
# python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name L2P \
#     --prompt_param 30 20 -1 -1  \
#     --log_dir ${OUTDIR}/vit/l2p

# dual prompt
# python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 10 20 6 -1 \
#     --log_dir ${OUTDIR}/vit/dual-prompt


###############
#    OLDIES   #
###############
    
# # # linear only
# # python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
# #     --learner_type prompt --learner_name Linear \
# #     --log_dir ${OUTDIR}/vit/linear

# # # LWF
# # python -u run.py --config $CONFIG_VIT --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
# #     --learner_type kd --learner_name LWF_MC \
# #     --log_dir ${OUTDIR}/vit/lwf-mc

# # # clip
if [ $# -eq 3 ]
    then
        python -u run.py --config $CONFIG_CLIP --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
            --learner_type prompt --learner_name L2P --prompt_param 30 20 -1 -1  --log_dir ${OUTDIR}/vit/clip $2 --template_style $3
elif [ $# -eq 2 ]
    then
        python -u run.py --config $CONFIG_CLIP --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
            --learner_type prompt --learner_name L2P --prompt_param 30 20 -1 -1  --log_dir ${OUTDIR}/vit/clip $2
else
    python -u run.py --config $CONFIG_CLIP --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
            --learner_type prompt --learner_name L2P --prompt_param 30 20 -1 -1  --log_dir ${OUTDIR}/vit/clip
fi

## example of last two arguments —freeze_last onlyprompt gibberish_template