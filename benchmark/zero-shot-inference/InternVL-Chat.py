import os
import random
import re
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

import argparse
import os.path as osp
import json
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def parse_evidence(evidence_text):
    pattern = r'(\d+)s\s+to\s+(\d+)s'
    matches = re.findall(pattern, evidence_text)
    
    time_intervals = []
    for match in matches:
        start_time = match[0]
        end_time = match[1]
        time_intervals.append([int(start_time), int(end_time)])
    
    if len(time_intervals) == 1:
        time_intervals = time_intervals[0]
    
    return time_intervals


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
# parser.add_argument("--gpu-id", default=0)
parser.add_argument("--gt-path", default='')
parser.add_argument("--video-root", default='')
parser.add_argument("--save-root", default='')
args = parser.parse_args()

device = torch.device(f"cuda:0")
dataset = VideoQATestset(args.gt_path)

path = 'InternVL2-8B/'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

generation_config = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)


results = []
success_num = 0
for sample in tqdm(dataset):
    question = sample['question']
    video_path = osp.join(args.video_root, sample['segment_id'] + '.mp4')

    pixel_values, num_patches_list = load_video(video_path, num_segments=30, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).to(device)
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    question = video_prefix + 'You are given an ego-centric video. Please watch the video and answer the following question: {}'.format(question.lower())
    # Frame1: <image>\nFrame2: <image>\n...\nFrame31: <image>\n{question}
    pred_answer, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list,
                                history=None, return_history=True)
    # print(f'User: {question}')
    # print(f'Assistant: {pred_answer}')

    # TODO: remove example
    question = 'The frames are sampled uniformly from 0s to 180s, namely at 6s, 12s, ..., 180s. Localize the time spans that semantically matches and support your answer of the question. For example, the answer can be deduced from 10s to 25s, and from 140s to 150s.'
    pred_evidence_text, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list,
                                history=history, return_history=True)
    # print(f'User: {question}')
    # print(f'Assistant: {pred_evidence_text}')

    try:
        pred_evidence = parse_evidence(pred_evidence_text)
        success_num += 1
    except:
        pred_evidence = sorted(random.sample(range(0, 180), 2)) #[0, 180]

    results.append({
        "sample_id": sample['sample_id'],
        "category": sample['category'],
        "Q": question,
        "A": pred_answer,
        "T": pred_evidence,
        "response": pred_answer + ' ' + pred_evidence_text, 
    })

print(f"Evidence parsing success rate: {success_num} / {len(results)}")
with open(os.path.join(args.save_root, "pred_internvl.json"), 'w') as f:
    json.dump(results, f, indent=2)
