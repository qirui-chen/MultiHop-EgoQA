# Adapted from https://github.com/NVlabs/LITA. Below is the original copyright:
# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import re
from collections import defaultdict
import glob
import numpy as np
np.set_printoptions(precision=2)

from torch.utils.data import Dataset, DataLoader

from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from gelm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from gelm.model.builder import load_pretrained_model
from gelm.utils import load_video_frames

    
# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, args, questions, tokenizer, image_processor, model_config):
        self.args = args
        self.questions = questions
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        vid = line["vid"]
        qs = line["question"]
        # if self.model_config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompt = prompt.strip()

        # load video frames
        # visual_path = os.path.join(self.args.image_folder, 'v_' + vid)
        # image_tensor = load_video_frames(visual_path, self.image_processor, self.model_config.num_frames)
        image_tensor = torch.load(os.path.join(self.args.image_folder, 'v_' + vid + '.pth.tar'), map_location="cpu")
        idx = np.round(np.linspace(0, image_tensor.shape[0] - 1, self.model_config.num_frames)).astype(int)
        image_tensor = image_tensor[idx]
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def parse_start_end_timestamps(outputs, duration, strict=False):
    timestamp_pattern = '\<(?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )\>'
    rx = re.compile(timestamp_pattern, re.VERBOSE)
    matches = list(rx.finditer(outputs))
    
    if strict:
        assert len(list(matches)) >= 2, "cannot find timestamps"
    elif len(list(matches)) < 2:
        return outputs, [0, duration]
    
    prev_end = 0
    sentence = ""
    timestamps = []
    for i in range(2):
        m = matches[i]
        start = m.start(0)
        end = m.end(0)
        timestamp = float(m.group(0)[1:-1])
        timestamp = min(max(timestamp, 0), duration)
        timestamps.append(timestamp)
        sentence += outputs[prev_end:start]
        prev_end = end
    sentence += outputs[prev_end:]
    sentence = sentence.strip()
    
    return sentence, [min(timestamps), max(timestamps)]



def alternate_replace(input_str, replacements):

    pattern = re.compile(r'<(\d+(\.\d+)?)>')
    replacement_index = 0

    def replace(match):
        nonlocal replacement_index
        # Get the current replacement value
        replacement_value = replacements[replacement_index]
        # Alternate the replacement index
        replacement_index = (replacement_index + 1) % len(replacements)
        # Return the replacement value inside angle brackets
        return f'<{replacement_value}>'

    # Perform the replacement using the sub method
    result = pattern.sub(replace, input_str)
    
    return result


def iou(seg1, seg2):
    assert seg1[1] >= seg1[0] and seg2[1] >= seg2[0]
    
    x1 = max(seg1[0], seg2[0])
    x2 = min(seg1[1], seg2[1])
    inter = max(x2 - x1, 0)
    
    len1 = max(seg1[1] - seg1[0], 0)
    len2 = max(seg2[1] - seg2[0], 0)
    
    union = len1 + len2 - inter
    
    if union == 0:
        return 0.0
    else:
        return inter/union
    

def precision_func(thres):
    def precision(seg1, seg2):
        return float(iou(seg1, seg2) >= thres)
    return precision


def eval_model(args):
    # metrics
    metric_func = {
        # 'iou': iou,
        # 'precision@0.5': precision_func(0.5),
        'saliency_iou': iou,
        'saliency_precision@0.5': precision_func(0.5),
        'similarity_iou': iou,
        'similarity_precision@0.5': precision_func(0.5)
    }
    metrics = {}
    for metric in metric_func:
        metrics[metric] = defaultdict(list)
        
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    assert not model.config.mm_use_im_start_end and not model.config.mm_use_im_patch_token, "not supported yet"

    if args.conv_mode is None:
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "tinyllama" in model.name.lower():
            conv_mode = "tiny_llama"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        args.conv_mode = conv_mode
        
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))[0:]
    data_loader = create_data_loader(args, questions, tokenizer, image_processor, model.config)
    out_list = []
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):

        stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style not in [SeparatorStyle.TWO, SeparatorStyle.TINY_LLAMA] else conv_templates[args.conv_mode].sep2
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids, saliency_proposal, similarity_proposal = model.generate_with_time(
                input_ids,
                images=image_tensor.to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
                multi_hop=False, #NOTE: because the official metric func support only one proposal!!
                factor_saliency=0.75, 
                factor_similarity=0.1, 
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        output_ids = output_ids[0, input_token_len:]

        duration = line['duration']
        saliency_timestamps = np.round(np.array(saliency_proposal) * duration / (model.config.num_frames - 1), decimals=2).tolist()
        similarity_timestamps = np.round(np.array(similarity_proposal) * duration / (model.config.num_frames - 1), decimals=2).tolist()
        
        outputs = tokenizer.batch_decode([output_ids], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # split predicted time stamps and explanation
        outputs = alternate_replace(outputs[len("<T1> </T1> ."):], similarity_timestamps)
        sentence, timestamps = parse_start_end_timestamps(outputs, duration, strict=False)
        sentence_ref, timestamps_ref = parse_start_end_timestamps(line['answer'], duration, strict=True)
        
        # eval timestamps
        for metric in metrics:
            timestamps_pred = timestamps
            if 'saliency_' in metric:
                timestamps_pred = saliency_timestamps
            if 'similarity_' in metric:
                timestamps_pred = similarity_timestamps
            metrics[metric][line['vid']].append(metric_func[metric](timestamps_pred, timestamps_ref))
        
        # save explanation
        ans_id = shortuuid.uuid()
        out = {"question_id": line["question_id"],
               "prompt": line["question"],
               "text_out": sentence,
               "text_gnd": sentence_ref,
            #    "time_text": timestamps,
               "time_saliency": saliency_timestamps,
               "time_similarity": similarity_timestamps,
               "time_gnd": timestamps_ref,
               "duration": duration,
               "similarity_iou": metric_func["similarity_iou"](similarity_timestamps, timestamps_ref),
               "similarity_precision@0.5": metric_func["similarity_precision@0.5"](similarity_timestamps, timestamps_ref),
               "answer_id": line["question_id"],
            #    "model_id": model_name,
            #    "metadata": {}
            }
        out_list.append(out)
    # exit(0)
    # summarize metrics
    averages = {}
    for metric in metrics:
        avg = []
        for vid in metrics[metric]:
            avg.append(np.mean(metrics[metric][vid]))
        averages[metric] = np.mean(avg)
    print(averages)
    
    # save file
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    answers_file = os.path.join(output_dir, 'answers.json')
    metrics_file = os.path.join(output_dir, 'metrics.json')
    
    with open(answers_file, 'w') as f:
        json.dump(out_list, f, indent=2)
        
    with open(metrics_file, 'w') as f:
        json.dump(averages, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--visual-data-type", type=str, default="video_frames")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
