# bash experiments/imagenet-r.sh
# experiment settings\\

## ABLATING KD !!!
SPLIT=10
DATASET=IMBALANCEINR
N_CLASS=200


# hard coded inputs
GPUID='0 1'
CONFIG_CLIP_P=configs/imnet-r_clip_prompt.yaml
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



MU=0
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

DATE=hepco_v4.0_iid_cutoff_kd_replay_1
OUTDIR=_outputs/${DATE}/${DATASET}/${SPLIT}-task
mkdir -p $OUTDIR

python -u run.py --config $CONFIG_CLIP_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name L2P \
    --prompt_param 100 20 1 -1 --kl --hepco --imbalance 1 --percent 0.1 --n_clients 4 --n_rounds 8 --cutoff --replay_ratio 1 --wandb_name $DATE \
    --log_dir ${OUTDIR}/vit/l2p_multi-layer

DATE=hepco_v4.0_iid_cutoff_kd_replay_0.5
OUTDIR=_outputs/${DATE}/${DATASET}/${SPLIT}-task
mkdir -p $OUTDIR

python -u run.py --config $CONFIG_CLIP_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name L2P \
    --prompt_param 100 20 1 -1 --kl --hepco --imbalance 1 --percent 0.1 --n_clients 4 --n_rounds 8 --cutoff --replay_ratio 0.5 --wandb_name $DATE \
    --log_dir ${OUTDIR}/vit/l2p_multi-layer

DATE=hepco_v4.0_iid_cutoff_kd_replay_0.75
OUTDIR=_outputs/${DATE}/${DATASET}/${SPLIT}-task
mkdir -p $OUTDIR

python -u run.py --config $CONFIG_CLIP_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name L2P \
    --prompt_param 100 20 1 -1 --kl --hepco --imbalance 1 --percent 0.1 --n_clients 4 --n_rounds 8 --cutoff --replay_ratio 0.75 --wandb_name $DATE \
    --log_dir ${OUTDIR}/vit/l2p_multi-layer

DATE=hepco_v4.0_iid_cutoff_kd_replay_0.25
OUTDIR=_outputs/${DATE}/${DATASET}/${SPLIT}-task
mkdir -p $OUTDIR

python -u run.py --config $CONFIG_CLIP_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name L2P \
    --prompt_param 100 20 1 -1 --kl --hepco --imbalance 1 --percent 0.1 --n_clients 4 --n_rounds 8 --cutoff --replay_ratio 0.25 --wandb_name $DATE \
    --log_dir ${OUTDIR}/vit/l2p_multi-layer

DATE=hepco_v4.0_iid_cutoff_kd_epoch_50
OUTDIR=_outputs/${DATE}/${DATASET}/${SPLIT}-task
mkdir -p $OUTDIR

python -u run.py --config $CONFIG_CLIP_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name L2P \
    --prompt_param 100 20 1 -1 --kl --hepco --imbalance 1 --percent 0.1 --n_clients 4 --n_rounds 8 --cutoff --kd_epochs 50 --wandb_name $DATE \
    --log_dir ${OUTDIR}/vit/l2p_multi-layer

DATE=hepco_v4.0_iid_cutoff_kd_epoch_100
OUTDIR=_outputs/${DATE}/${DATASET}/${SPLIT}-task
mkdir -p $OUTDIR

python -u run.py --config $CONFIG_CLIP_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name L2P \
    --prompt_param 100 20 1 -1 --kl --hepco --imbalance 1 --percent 0.1 --n_clients 4 --n_rounds 8 --cutoff --kd_epochs 100 --wandb_name $DATE \
    --log_dir ${OUTDIR}/vit/l2p_multi-layer

DATE=hepco_v4.0_iid_cutoff_kd_epoch_300
OUTDIR=_outputs/${DATE}/${DATASET}/${SPLIT}-task
mkdir -p $OUTDIR

python -u run.py --config $CONFIG_CLIP_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name L2P \
    --prompt_param 100 20 1 -1 --kl --hepco --imbalance 1 --percent 0.1 --n_clients 4 --n_rounds 8 --cutoff --kd_epochs 300 --wandb_name $DATE \
    --log_dir ${OUTDIR}/vit/l2p_multi-layer

DATE=hepco_v4.0_iid_cutoff_kd_epoch_400
OUTDIR=_outputs/${DATE}/${DATASET}/${SPLIT}-task
mkdir -p $OUTDIR

python -u run.py --config $CONFIG_CLIP_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name L2P \
    --prompt_param 100 20 1 -1 --kl --hepco --imbalance 1 --percent 0.1 --n_clients 4 --n_rounds 8 --cutoff --kd_epochs 400 --wandb_name $DATE \
    --log_dir ${OUTDIR}/vit/l2p_multi-layer

DATE=hepco_v4.0_iid_cutoff_kd_lr_0.001
OUTDIR=_outputs/${DATE}/${DATASET}/${SPLIT}-task
mkdir -p $OUTDIR

python -u run.py --config $CONFIG_CLIP_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name L2P \
    --prompt_param 100 20 1 -1 --kl --hepco --imbalance 1 --percent 0.1 --n_clients 4 --n_rounds 8 --cutoff --kd_lr 1e-3 --wandb_name $DATE \
    --log_dir ${OUTDIR}/vit/l2p_multi-layer

DATE=hepco_v4.0_iid_cutoff_kd_lr_0.005
OUTDIR=_outputs/${DATE}/${DATASET}/${SPLIT}-task
mkdir -p $OUTDIR

python -u run.py --config $CONFIG_CLIP_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name L2P \
    --prompt_param 100 20 1 -1 --kl --hepco --imbalance 1 --percent 0.1 --n_clients 4 --n_rounds 8 --cutoff --kd_lr 5e-3 --wandb_name $DATE \
    --log_dir ${OUTDIR}/vit/l2p_multi-layer



# l2p
# python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name L2P --mu $MU \
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

# # g
# python -u run.py --config $CONFIG_CLIP_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type prompt --learner_name L2P --mu $MU \
#     --prompt_param 30 20 -1 -1  \
#     --log_dir ${OUTDIR}/vit/clip
    #--learner_type default --learner_name NormalNN \
    
## clip l2p multilayer (freeze last)
# python -u run.py --config $CONFIG_CLIP_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#      --learner_type prompt --learner_name L2P --mu $MU --log_dir ${OUTDIR}/clip/l2p_multilayer \
#     --prompt_param 100 20 1 -1 --freeze_last

## clip dualprompt (freeze last)
# python -u run.py --config $CONFIG_CLIP_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#      --learner_type prompt --learner_name DualPrompt --mu $MU --log_dir ${OUTDIR}/clip/dualprompt \
#     --prompt_param 10 20 6 -1 --freeze_last

# # # clip ZS with prompting
# python -u run.py --config $CONFIG_CLIP_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#     --learner_type default --learner_name NormalNN --log_dir ${OUTDIR}/vit/clip --only_eval_zs \