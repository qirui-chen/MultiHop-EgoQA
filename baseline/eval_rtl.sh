MODEL_NAME=RTL-GeLM-7B


CUDA_VISIBLE_DEVICES=7 python gelm/eval/inference_rtl.py \
    --model-path finetuned/$MODEL_NAME  \
    --question-file datasets/temporal_reasoning/annot_val_1_q229.json \
    --image-folder datasets/activitynet-captions/intern_feature \
    --output-dir outputs/RTL/${MODEL_NAME} \
    --conv-mode v1 \

# OPENAI_API_KEY="" python gelm/eval/eval_gpt_review_rtl.py \
#     --context datasets/activitynet-captions/val_1.json \
#     --answer \
#     outputs/RTL/${MODEL_NAME}/answers.json \
#     --rule gelm/eval/table/rule.txt \
#     --output outputs/RTL/${MODEL_NAME}/review.jsonl

# python gelm/eval/summarize_gpt_review.py -f outputs/RTL/${MODEL_NAME}/review.jsonl

