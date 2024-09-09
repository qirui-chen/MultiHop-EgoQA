MODEL_NAME=EgoQA-GeLM-7B
QUESTION_FILE=path/to/MultiHop-EgoQA.json

CUDA_VISIBLE_DEVICES=7 python gelm/eval/inference_multihop_qa.py \
    --model-path finetuned/$MODEL_NAME  \
    --question-file $QUESTION_FILE \
    --image-folder datasets/multihop_qa/features \
    --output-dir outputs/MultiHopQA/$MODEL_NAME \
    --conv-mode v1 \
