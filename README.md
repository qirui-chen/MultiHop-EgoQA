# Grounded Multi-Hop VideoQA in Long-Form Egocentric Videos

[🏡 Project Page](qirui-chen.github.io/MultiHop-EgoQA) |  [📄 Paper]() | [🤗 Dataset](https://huggingface.co/datasets/SurplusDeficit/MultiHop-EgoQA)



## Abstract

### Problem Scenario
This paper considers the problem of <em>Multi-Hop Video Question Answering (<u>MH-VidQA</u>)</em> in long-form egocentric videos. This task not only requires to answer visual questions, but also to localize multiple relevant time intervals within the video as visual evidences.

<div align="center">
   <img src="./assets/teaser2.jpeg" style="width: 80%;">
</div>

### Baseline Method

We develop an automated pipeline to mine multi-hop question-answering pairs with associated temporal evidence, enabling to construct a large-scale dataset for instruction-tuning. We then propose a novel architecture, termed as <b><u>GeLM</u></b>, to leverage the world knowledge reasoning capabilities of multi-modal large language models (LLMs), while incorporating a grounding module to retrieve temporal evidence in the video with flexible grounding tokens.

<div align="center">
   <img src="./assets/architecture_v3.jpeg" style="width: 100%;">
</div>


## TODO

- [ ] Release the annotation file
- [ ] Release codes of the evaluation
- [ ] Release codes of the baseline method
- [ ] Release automatically constructed data
- [ ] Release visual features


## 🫡 Acknowledgements

- Our baseline method implementation is adapted from the [LITA](https://github.com/NVlabs/LITA).

- The implementation of the zero-shot evaluation code references the official repositories of [TimeChat](https://github.com/RenShuhuai-Andy/TimeChat) and [VTimeLLM](https://github.com/huangb23/VTimeLLM), as well as the Hugging Face documentation of [InternVL2](https://huggingface.co/OpenGVLab/InternVL2-8B), [LLaVa-NeXT-Video](https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B), [LLaVa-v1.6](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf), [Meta-Llama-3.1](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), and the documentation of OpenAI [GPT-4o](https://github.com/openai/openai-cookbook/blob/main/examples/GPT_with_vision_for_video_understanding.ipynb) for video understanding.