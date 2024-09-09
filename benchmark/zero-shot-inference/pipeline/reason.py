import argparse
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = """Here is a video shot from a first-person perspective, recording my activities. I will provide descriptions of each second of the video. Your task is to answer the question I give you based on these video descriptions. Ensure your answer is concise. After answering, provide the time intervals that related to the question-answering as evidence. Evidence is list of intervals (e.g., [[s1, e1], ..., [s_i, e_i], ...]) and each time interval (e.g., [s_i, e_i], s_i <= e_i) consists of one start timestamp and one end timestamp. Note that evidence may include multiple time intervals. Finally, explain your answer in one sentence.

Here is an example format for your response.
###Question
When did I open the tap?

###Answer
You opened the tap from 10s to 20s and 60s to 70s.

###Evidence
[[10, 20], [60, 70]]

###Rationale
According to the action descriptions with timestamps, you opened the tap for two times from 10s to 20s and from 60s to 70s.

Here is the video and question you need to review:
###Captions
timestamp, caption
{}

###Question
{}

Your response should strictly follow the format of given example, including three parts: ###Answer, ###Evidence, and ###Rationale. Do not add any extra content.
"""


class VideoQATestset(Dataset):

    def __init__(self, gt_path, captions_path):
        with open(gt_path, 'r') as f:
            self.gt_data = json.load(f)

        with open(captions_path, 'r') as f:
            self.captions_data = json.load(f)

    def __len__(self):
        return len(self.gt_data)

    def __getitem__(self, idx):
        gt_item = self.gt_data[idx]
        segment_id = gt_item['segment_id']
        caption = self.captions_data.get(segment_id, "")
        caption_texts = '\n'.join([' '.join([str(t), text]) for t, text in caption])

        return {
            'segment_id': gt_item['segment_id'],
            'sample_id': gt_item['sample_id'],
            'category': gt_item['category'],
            'question': gt_item['Q'],
            'caption': caption_texts
        }


def merge_intervals(spans, gap=8):
    """
    1. connect <gap-s non-overlap
    2. union overlap
    """
    intervals = spans.copy()
    if type(intervals[0]) == int:
        return intervals
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        previous = merged[-1]
        if current[0] <= previous[1]:
            merged[-1] = [previous[0], max(previous[1], current[1])]
        elif current[0] - previous[1] < gap:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)

    if len(merged) == 1 and type(merged[0]) == list:
        merged = merged[0]

    return merged


def extract_sections(text):
    answer_text = re.search(r"###Answer\s*(.*?)\s*(?=###|$)", text, re.DOTALL)
    evidence_text = re.search(r"###Evidence\s*(.*?)\s*(?=###|$)", text, re.DOTALL)
    rationale_text = re.search(r"###Rationale\s*(.*?)\s*(?=###|$)", text, re.DOTALL)

    answer_text = answer_text.group(1).strip() if answer_text else ""
    evidence_text = evidence_text.group(1).strip() if evidence_text else ""
    rationale_text = rationale_text.group(1).strip() if rationale_text else ""

    return answer_text, evidence_text, rationale_text


def parse_evidence(evidence_text):
    evidence = eval(evidence_text)
    assert is_intervals(evidence), evidence_text
    return evidence


def is_intervals(data):

    def is_interval(interval):
        return isinstance(interval, list) and len(interval) == 2 and isinstance(interval[0], int) and isinstance(
            interval[1], int) and interval[0] <= interval[1]

    if is_interval(data):
        return True  #"single interval"
    elif isinstance(data, list) and all(is_interval(interval) for interval in data):
        return True  #"list of intervals"
    else:
        return False  #"neither"


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt-path',
        type=str,
        default="",
    )
    parser.add_argument('--captions-path', type=str, default="")
    parser.add_argument('--save-root', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inference')
    args = parser.parse_args()

    # Load dataset
    gt_path = args.gt_path
    captions_path = args.captions_path
    dataset = VideoQATestset(gt_path, captions_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model_id = "Meta-Llama-3.1-8B-Instruct/"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    results = []
    success_num = 0
    pbar = tqdm(dataloader, total=len(dataset))
    for batch in pbar:
        captions = batch['caption']
        sample_ids = batch['sample_id']
        categories = batch['category']
        questions = batch['question']

        messages = [[{
            "role": "user",
            "content": PROMPT.format(caption, question)
        }] for caption, question in zip(captions, questions)]

        input_ids = tokenizer.apply_chat_template(messages,
                                                  add_generation_prompt=True,
                                                  padding=True,
                                                  return_tensors="pt").to(model.device)

        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

        #TODO: regex & special token
        for i in range(outputs.shape[0]):
            response = outputs[i][input_ids.shape[-1]:]
            response_text = tokenizer.decode(response, skip_special_tokens=True)
            # print(response_text)

            answer, evidence_text, rationale = extract_sections(response_text)
            try:
                evidence = parse_evidence(evidence_text)
                evidence = merge_intervals(evidence)
                success_num += 1
            except:
                evidence = sorted(random.sample(range(0, 180), 2))  #[0, 180]

            results.append({
                "sample_id": sample_ids[i],
                "category": categories[i],
                "Q": questions[i],
                "A": answer,
                "T": evidence,
                "response": response_text,
            })
        pbar.set_postfix({"success_num": success_num})
        pbar.update(outputs.shape[0])

    pbar.close()
    with open(os.path.join(args.save_root, "pred_pipeline_3.1_8B.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Success rate: {success_num} / {len(results)}")
