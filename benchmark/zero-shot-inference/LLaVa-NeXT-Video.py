import os
import av
import torch
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

import argparse
import os.path as osp
import json
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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

parser = argparse.ArgumentParser(description="")
parser.add_argument("--gt-path", default='')
parser.add_argument("--video-root", default='')
parser.add_argument("--save-root", default='')
args = parser.parse_args()

device = torch.device(f"cuda:0")
dataset = VideoQATestset(args.gt_path)

model_id = "LLaVA-NeXT-Video-7B-hf"
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(device)

processor = LlavaNextVideoProcessor.from_pretrained(model_id)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


results = []
for sample in tqdm(dataset):
    question = sample['question']
    video_path = osp.join(args.video_root, sample['segment_id'] + '.mp4')

    # define a chat histiry and use `apply_chat_template` to get correctly formatted prompt1
    # Each value in "content" has to be a list of dicts with types ("text", "image", "video") 
    conv = [
        {

            "role": "user",
            "content": [
                {"type": "text", "text": "You are given frames of a video. According to the input frames, answer the following question concisely: {}.".format(question.lower())},
                {"type": "video"},
                ],
        },
    ]
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)

    container = av.open(video_path)

    # sample uniformly # frames from the video, can sample more for longer videos
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 32).astype(int)
    clip = read_video_pyav(container, indices)
    inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

    output = model.generate(**inputs_video, max_new_tokens=256, do_sample=False)
    output_text = processor.decode(output[0], skip_special_tokens=True)
    pred_answer = output_text.split('ASSISTANT:')[-1].strip()
    # print(pred_answer)

    results.append({
        "sample_id": sample['sample_id'],
        "category": sample['category'],
        "Q": question,
        "A": pred_answer,
        # "response": response, 
    })

with open(os.path.join(args.save_root, "pred_llavanext.json"), 'w') as f:
    json.dump(results, f, indent=2)
