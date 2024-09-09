# Adapted from https://github.com/NVlabs/LITA. Below is the original copyright:
# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE

import os
import glob
import json
import numpy as np
import re
import torch

from gelm.dataset.base_dataset import BaseDataset
from gelm.constants import DEFAULT_IMAGE_TOKEN


class TemporalReasoningDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(TemporalReasoningDataset, self).__init__(data_path, tokenizer, data_args)
        
    def get_sources(self, i):
        vqas = self.list_data_dict[i]
        return self.format_temporal_reasoning(vqas)
    
    def get_visual(self, sources):
        if self.visual_data_type == 'video_frames':
            return self.load_video_frames(sources['image'])
        elif self.visual_data_type == 'video':
            return self.load_video(sources['image'], self.data_args.num_frames)
        elif self.visual_data_type == 'feature':
            feature = torch.load(sources['image'], map_location="cpu")
            idx = np.round(np.linspace(0, feature.shape[0] - 1, self.data_args.num_frames)).astype(int)
            return feature[idx]
        
    def format_temporal_reasoning(self, vqas):
        out = {}
        vid = vqas['id']
        out['id'] = vid
        
        if self.visual_data_type == 'video_frames':
            frames = sorted(glob.glob(os.path.join(self.image_folder, vid, '*'+ self.ext)))
            idx = np.round(np.linspace(0, len(frames) - 1, self.data_args.num_frames)).astype(int)
            out['image'] = list(np.array(frames)[idx])
        elif self.visual_data_type == 'video':
            out['image'] = os.path.join(self.image_folder, captions['image'])
        elif self.visual_data_type == 'feature':
            out['image'] = os.path.join(self.image_folder, vid + '.pth.tar')
            
        convo = []
        duration = vqas['duration']

        for i, vqa in enumerate(vqas['QA']):
            if i == 0:
                gpt_prompt = DEFAULT_IMAGE_TOKEN + '\n'
            else:
                gpt_prompt = ""
                
            question = vqa['q']
            answer = vqa['a']
            
            gpt_prompt += question.strip()
            
            # process answer
            timestamp_pattern = '\<(?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )\>'
            rx = re.compile(timestamp_pattern, re.VERBOSE)
            new_answer = "<T1></T1>."
            
            interval = []
            for m in rx.finditer(answer):
                timestamp = float(m.group(0)[1:-1])
                if len(interval) < 2:
                    interval.append(int(np.round((self.data_args.num_frames-1) * (timestamp / duration))))

            new_answer = "<T1></T1>." + answer
          
            gpt_value = new_answer.strip()
            convo.append({"from": "human", "value": gpt_prompt.strip()})
            convo.append({"from": "gpt", "value": gpt_value.strip()})

        out['saliency'] = torch.zeros(self.data_args.num_frames)

        if len(interval) == 2 and interval[1] >= interval[0]:
            out['saliency'][interval[0]:interval[1]+1] = 1

        out['evidence'] = out['saliency'].unsqueeze(0)

        out['conversations'] = convo
        
        return out
                
                
class TemporalReasoningDataset_activitynet(TemporalReasoningDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(TemporalReasoningDataset_activitynet, self).__init__(data_path, tokenizer, data_args)
    
    def set_params(self):
        self.image_folder = os.path.join(self.data_path, 'activitynet-captions', 'intern_feature') 
        self.feature_list = os.listdir(self.image_folder)
        self.visual_data_type = 'feature'

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'temporal_reasoning', 'activitynet_train_gpt-4-0613_temp_6_f10009.json')
        data_dict = json.load(open(data_path, "r"))
        for vid in data_dict:
            if vid + '.pth.tar' not in self.feature_list:
                continue
            data = data_dict[vid]            
            for vqa in data['QA']:
                out = {}
                out['id'] = vid
                out['duration'] = data['duration']
                out['QA'] = [vqa]

                answer = vqa['a']
                timestamp_pattern = '\<(?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )\>'
                rx = re.compile(timestamp_pattern, re.VERBOSE)
                interval = []
                for m in rx.finditer(answer):
                    timestamp = float(m.group(0)[1:-1])
                    if len(interval) < 2:
                        interval.append(timestamp)
                if len(interval) == 2 and interval[1] >= interval[0]:
                    self.list_data_dict.append(out)

        print("# Samples: {}".format(len(self.list_data_dict)))
            