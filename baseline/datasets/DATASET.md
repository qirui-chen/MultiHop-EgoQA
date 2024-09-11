# Preparing Datasets for Grounded VideoQA


## MultiHop-EgoQA

- Raw videos are available for download under the [Ego4D](https://ego4d-data.org/docs/start-here/) license.

- Download the annotation data (question-answer-evidence triplets) and video features from [Hugging Face](https://huggingface.co/datasets/SurplusDeficit/MultiHop-EgoQA).

- After downloading, save the annotation file `MultiHop-EgoQA.json` under `benchmark/metrics/`, and video features under `baseline/datasets/multihop_qa/features`. 



## ActivityNet-RTL

- For annotation data, please refer to the document of [ActivityNet-RTL](https://github.com/NVlabs/LITA/blob/main/docs/Video_Data.md).

- We provide InternVideo visual features of the ActivityNet in the same repo from [Hugging Face](https://huggingface.co/datasets/SurplusDeficit/MultiHop-EgoQA).

- After downloading, put the features under `baseline/datasets/activitynet-captions/intern_feature`.