import argparse
import base64
import json
import os
import os.path as osp
import random
import re
import time
from itertools import chain

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import requests
from openai import OpenAI
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROMPT = """Here is a 3-minute egocentric video recording my activities. I will provide you with frames sampled every 3 seconds from the video. Your task is to answer a specific question based on these frames accurately. Ensure your answer is concise.

After answering, list the time intervals related to the question as evidence. Each interval should have a start and end timestamp in seconds (e.g., [[9, 15], [120, 135]]). Note that the duration of the video is 180s and the frames are sampled at 0s, 3s, 6s, ..., 180s.

Finally, explain your answer in one sentence.

Here's an example of the response format:

### Question:
How many times did I open the tap?

### Frames:
The frame sampled at 0s:
......

### Answer:
You opened the tap twice.

### Evidence:
[[9, 15], [120, 135]]

### Rationale:
According to the frames sampled from 9s to 15s, and from 120s to 135s, you opened the tap twice.

Your response should strictly follow the example format, including three parts: Answer, Evidence, and Rationale. Do not add any extra content. Here is the question you need to answer:

### Question:
{}

### Frames:"""


class VideoQATestset(Dataset):

    def __init__(self, gt_path):
        with open(gt_path, 'r') as f:
            self.gt_data = json.load(f)

    def __len__(self):
        return len(self.gt_data)

    def __getitem__(self, idx):
        gt_item = self.gt_data[idx]

        return {
            'segment_id': gt_item['segment_id'],
            'sample_id': gt_item['sample_id'],
            'category': gt_item['category'],
            'question': gt_item['Q'],
        }


def extract_sections(text):
    answer_text = re.search(r"Answer:?\s*(.*?)\s*(?=###|$)", text, re.DOTALL)
    evidence_text = re.search(r"Evidence:?\s*(.*?)\s*(?=###|$)", text, re.DOTALL)
    rationale_text = re.search(r"Rationale:?\s*(.*?)\s*(?=###|$)", text, re.DOTALL)

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt-path',
        type=str,
        default="",
    )
    parser.add_argument(
        '--video-root',
        type=str,
        default="",
    )
    parser.add_argument(
        '--save-root',
        type=str,
        default="",
    )
    args = parser.parse_args()

    dataset = VideoQATestset(args.gt_path)
    results = []
    success_num = 0
    for sample in tqdm(dataset):
        question = sample['question']
        video_path = osp.join(args.video_root, sample['segment_id'] + '.mp4')

        video = cv2.VideoCapture(video_path)

        base64Frames = []
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        video.release()

        PROMPT_MESSAGES = [
            {
                "role":
                    "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT.format(question)
                    },
                    # 180s * 30FPS = 5400 | 90 frames / 30 FPS = 3s
                    *chain.from_iterable(
                        zip(({
                            "type": "text",
                            "text": f"\nThe frame sampled at {x}s:"
                        } for x in range(0,
                                         len(base64Frames[0::90]) * 3, 3)), ({
                                             "type": "image_url",
                                             "image_url": {
                                                 "url": f"data:image/jpeg;base64,{x}",
                                                 "detail": "low"
                                             }
                                         } for x in base64Frames[0::90])))
                ],
            },
        ]

        params = {
            "model": "gpt-4o-2024-05-13",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 256,
        }

        client = OpenAI(base_url="", api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(**params)

        # print(response.choices[0].message.content)
        response_text = response.choices[0].message.content
        # print(response_text)

        answer, evidence_text, rationale = extract_sections(response_text)
        try:
            evidence = parse_evidence(evidence_text)
            evidence = merge_intervals(evidence)
            success_num += 1
        except:
            evidence = sorted(random.sample(range(0, 180), 2))  #[0, 180]

        results.append({
            "sample_id": sample['sample_id'],
            "category": sample['category'],
            "Q": question,
            "A": answer,
            "T": evidence,
            "response": response_text,
        })

    print(f"Evidence parsing success rate: {success_num} / {len(results)}")
    with open(os.path.join(args.save_root, "pred_gpt-4o.json"), 'w') as f:
        json.dump(results, f, indent=2)
