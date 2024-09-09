import os
import random
import re
import sys
import argparse
import torch
from vtimellm.constants import IMAGE_TOKEN_INDEX
from vtimellm.conversation import conv_templates, SeparatorStyle
from vtimellm.model.builder import load_pretrained_model, load_lora
from vtimellm.utils import disable_torch_init
from vtimellm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor
from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
from easydict import EasyDict as edict
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import numpy as np
import clip

import os.path as osp
import json
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def inference(model, image, query, tokenizer):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image[None,].cuda(),
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=256,
            use_cache=True)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


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

def parse_evidence(evidence_text):

    time_intervals = []
    pattern = r"(\d+)"
    matches = re.findall(pattern, evidence_text)
    for percentages in matches[:2]:
        time_intervals.append(int((int(percentages) / 100) * 180))

    assert len(time_intervals) == 2
    
    return time_intervals


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--clip_path", type=str, default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--model_base", type=str, default="vicuna-7b-v1.5")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage3")
    parser.add_argument("--gt-path", default='')
    parser.add_argument("--video-root", default='')
    parser.add_argument("--save-root", default='')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)
    model = model.cuda()
    # model.get_model().mm_projector.to(torch.float16)
    model.to(torch.float16)

    clip_model, _ = clip.load(args.clip_path)
    clip_model.eval()
    clip_model = clip_model.cuda()

    video_loader = VideoExtractor(N=100)

    dataset = VideoQATestset(args.gt_path)
    results = []
    success_num = 0
    for sample in tqdm(dataset):
        question = sample['question']
        video_path = osp.join(args.video_root, sample['segment_id'] + '.mp4')

        _, images = video_loader.extract({'id': None, 'video': video_path})

        transform = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # print(images.shape) # <N, 3, H, W>
        images = transform(images / 255.0)
        images = images.to(torch.float16)
        with torch.no_grad():
            features = clip_model.encode_image(images.to('cuda'))

        query = question
        pred_answer = inference(model, features, "<video>\n " + query, tokenizer)
        # print("query: ", query)
        # print("answer: ", pred_answer)

        query = f"During which frames can we see '{pred_answer}' happening in the video?"
        pred_evidence_text = inference(model, features, "<video>\n " + query, tokenizer)
        # print("query: ", query)
        # print("answer: ", pred_evidence_text)

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
with open(osp.join(args.save_root, "pred_vtimellm.json"), 'w') as f:
    json.dump(results, f, indent=2)
