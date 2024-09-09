METHOD="baseline"
PREDICTION_ROOT="predictions"
GT_FILE="MultiHop-EgoQA.json"

mkdir -p results/${METHOD}

python evaluate_grounding.py \
    --pred_file ${PREDICTION_ROOT}/pred_${METHOD}.json \
    --gt_file $GT_FILE \
    --results_file results/${METHOD}/grounding.json

# python evaluate_answering.py \
#     --pred_file ${PREDICTION_ROOT}/pred_${METHOD}.json \
#     --gt_file $GT_FILE \
#     --results_file results/${METHOD}/answering.json > results/${METHOD}/summary.txt