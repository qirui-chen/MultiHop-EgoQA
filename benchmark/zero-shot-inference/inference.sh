GT_PATH="<PATH>/gt.json"
VIDEO_PATH="<VIDEO ROOT>"
SAVE_PATH="<PATH>/metrics/predictions_final"
CAPTIONS_PATH="<PATH>/test_captions.json"

mkdir -p $SAVE_PATH


# GPT-4o
python GPT-4o.py \
    --gt-path $GT_PATH \
    --video-root $VIDEO_PATH \
    --save-root $SAVE_PATH \


# Pipeline
# cd pipeline
# CUDA_VISIBLE_DEVICES=0 python reason.py \
#     --gt-path $GT_PATH \
#     --captions-path $CAPTIONS_PATH \
#     --save-root $SAVE_PATH \


# InternVL-2-Chat
# CUDA_VISIBLE_DEVICES=0 python InternVL-Chat.py \
#     --gt-path $GT_PATH \
#     --video-root $VIDEO_PATH \
#     --save-root $SAVE_PATH \

# LLaVa-NeXT-Video
# CUDA_VISIBLE_DEVICES=0 python LLaVa-NeXT-Video.py \
#     --gt-path $GT_PATH \
#     --video-root $VIDEO_PATH \
#     --save-root $SAVE_PATH \


# VTimeLLM
# cd VTimeLLM
# conda activate vtimellm
# CUDA_VISIBLE_DEVICES=0 python -m vtimellm.VTimeLLM \
#     --gt-path $GT_PATH \
#     --video-root $VIDEO_PATH \
#     --save-root $SAVE_PATH \

# TimeChat
# cd TimeChat
# conda activate timechat
# python TimeChat.py --gpu_id 0 \
#     --gt-path $GT_PATH \
#     --video-root $VIDEO_PATH \
#     --save-root $SAVE_PATH \

