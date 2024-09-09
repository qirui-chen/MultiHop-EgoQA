import argparse
import json
import os
import random
import re
import subprocess
import time

import cv2
import decord
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import torchshow as ts
from decord import VideoReader
from tqdm import tqdm

from timechat.common.config import Config
from timechat.common.dist_utils import get_rank
from timechat.common.registry import registry
from timechat.conversation.conversation_video import (Chat, Conversation,
                                                      SeparatorStyle,
                                                      conv_llava_llama_2,
                                                      default_conversation)
from timechat.processors.video_processor import ToTHWC, ToUint8, load_video

decord.bridge.set_bridge('torch')

import random as rnd

import gradio as gr
from PIL import Image
# imports modules for registration
from timechat.datasets.builders import *
from timechat.models import *
from timechat.processors import *
from timechat.runners import *
from timechat.tasks import *
from transformers import StoppingCriteria, StoppingCriteriaList



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


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/timechat.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--video-root", default='')
    parser.add_argument("--gt-path", default='')
    parser.add_argument("--save-root", default='')

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args(args=[])
    return args


print('Initializing Chat')
args = parse_args()
cfg = Config(args)

DIR = "ckpt/timechat"
MODEL_DIR = f"{DIR}/timechat_7b.pth"

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_config.ckpt = MODEL_DIR
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()

vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))

dataset = VideoQATestset(args.gt_path)


results = []
success_num = 0
for sample in tqdm(dataset):
    question = sample['question']

    img_list = []
    # Initialize conv 
    chat_state = conv_llava_llama_2.copy()
    chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."

    try:
        msg = chat.upload_video_without_audio(
            video_path=os.path.join(args.video_root, sample['segment_id'] + '.mp4'),
            conv=chat_state,
            img_list=img_list,
            n_frms=96,
        )
    except decord._ffi.base.DECORDError:
        continue

    text_input = "You are given an ego-centric video. Please watch the video and answer the following question: '{}'".format(question.lower())
    chat.ask(text_input, chat_state)
    # print(text_input)
    pred_answer = chat.answer(conv=chat_state,
                            img_list=img_list,
                            num_beams=args.num_beams,
                            temperature=args.temperature,
                            max_new_tokens=300,
                            max_length=2000)[0]
    # print(pred_answer)

    text_input = "Detect and report the start and end timestamps of the video segment that semantically matches the textual query '{}'".format(pred_answer.lower())
    chat.ask(text_input, chat_state)
    # print(text_input)
    pred_evidence_text = chat.answer(conv=chat_state,
                            img_list=img_list,
                            num_beams=args.num_beams,
                            temperature=args.temperature,
                            max_new_tokens=300,
                            max_length=2000)[0]
    # print(pred_evidence_text)

    try:
        matches = re.findall(r'(\d+(\.\d+)?)\s*-\s*(\d+(\.\d+)?)', pred_evidence_text)
        pred_evidence = [[int(float(start)), int(float(end))] for start, _, end, _ in matches]
        if len(pred_evidence) == 1:
            pred_evidence = pred_evidence[0]
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

print(f"Success rate: {success_num} / {len(results)}")
with open(os.path.join(args.save_root, "pred_timechat.json"), 'w') as f:
    json.dump(results, f, indent=2)
