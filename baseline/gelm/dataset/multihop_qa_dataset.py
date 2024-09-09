import os
import glob
import json
import random
import numpy as np
import re
import torch
import torch.nn.functional as F

from gelm.dataset.base_dataset import BaseDataset
from gelm.constants import DEFAULT_IMAGE_TOKEN


def pad_mask_matrix(sequences):
    max_entities = max([s.size(0) for s in sequences])
    # Assume the last dimension are the same ([S, T] * B)

    out_tensor = []
    for mask_matrix in sequences:
        num_entities, num_frames = mask_matrix.shape[:]
        padded_mask_matrix = F.pad(mask_matrix, 
        (0, 0, 0, max_entities-num_entities),
        "constant",
        False
        )
        out_tensor.append(padded_mask_matrix)


class MultiHopQADataset(BaseDataset):

    def __init__(self, data_path, tokenizer, data_args):
        super(MultiHopQADataset, self).__init__(data_path, tokenizer, data_args)

    def get_sources(self, i):
        vqas = self.list_data_dict[i]
        return self.format_multihop_qa(vqas)

    def get_visual(self, sources):
        if self.visual_data_type == 'video_frames':
            return self.load_video_frames(sources['image'])
        elif self.visual_data_type == 'video':
            return self.load_video(sources['image'], self.data_args.num_frames)
        elif self.visual_data_type == 'feature':
            feature = torch.load(sources['image'], map_location="cpu")
            return feature

    def format_multihop_qa(self, vqas):
        out = {}
        vid = vqas['id']
        out['id'] = vid

        if self.visual_data_type == 'video_frames':
            frames = sorted(glob.glob(os.path.join(self.image_folder, vid, '*' + self.ext)))
            idx = np.round(np.linspace(0, len(frames) - 1, self.data_args.num_frames)).astype(int)
            out['image'] = list(np.array(frames)[idx])
        elif self.visual_data_type == 'video':
            out['image'] = os.path.join(self.image_folder, captions['image'])
        elif self.visual_data_type == 'feature':
            out['image'] = os.path.join(self.image_folder, vid + '.pth.tar')

        convo = []
        duration = vqas['duration']

        for i, vqa in enumerate(vqas['QA']):  # list of single-turn qa
            if i == 0:
                gpt_prompt = DEFAULT_IMAGE_TOKEN + '\n'
            else:
                gpt_prompt = ""

            question = vqa['Q']
            answer = vqa['A']

            gpt_prompt += question.strip()

            new_answer = answer # + '<T></T>'
            # new_answer = '<T></T>'

            gpt_value = new_answer.strip()
            convo.append({"from": "human", "value": gpt_prompt.strip()})
            convo.append({"from": "gpt", "value": gpt_value.strip()})

        out['saliency'] = torch.zeros(self.data_args.num_frames)
        intervals = [vqa['T']] if len(vqa['T']) == 2 and isinstance(vqa['T'][0], int) else vqa['T']
        for start, end in intervals:
            out['saliency'][start:end + 1] = 1

        # (#entities, T)
        out['evidence'] = []
        for entity_idx, intervals in vqa['Evidence'].items():
            evidence_mask = torch.zeros(self.data_args.num_frames)
            intervals = [intervals] if len(intervals) == 2 and isinstance(intervals[0], int) else intervals
            for start, end in intervals:
                evidence_mask[start:end + 1] = 1
            out['evidence'].append(evidence_mask)
        out['evidence'] = torch.stack(out['evidence'], dim=0)

        out['conversations'] = convo

        return out


class MultiHopQADataset_ego4d(MultiHopQADataset):

    def __init__(self, data_path, tokenizer, data_args):
        super(MultiHopQADataset_ego4d, self).__init__(data_path, tokenizer, data_args)

    def set_params(self):
        self.image_folder = os.path.join(self.data_path, 'multihop_qa', 'features')
        self.feature_list = os.listdir(self.image_folder)
        self.visual_data_type = 'feature'

    def ensure_format(self, intervals):
        is_intervals = True
        if len(intervals) == 0:
            is_intervals = False
            return is_intervals
        intervals = [intervals] if isinstance(intervals[0], int) else intervals
        if not all(
                isinstance(interval, list) and len(interval) == 2 and interval[0] <= interval[1]
                for interval in intervals):
            is_intervals = False
        return is_intervals

    def find_max_number_in_tokens(self, answer):
        tokens = re.findall(r'</?T(\d+)>', answer)
        numbers = [int(num) for num in tokens]
        return max(numbers) if numbers else 0

    def check_and_count_tags(self, s):
        pattern = re.compile(r'<(/?T\d+)>')
        tags = pattern.findall(s)
        
        stack = []
        count = 0
        matched_pairs = set()

        for tag in tags:
            if not tag.startswith('/'):
                stack.append(tag)
            else:
                if stack and stack[-1] == tag[1:]:
                    start_tag = stack.pop()
                    pair = (start_tag, tag)
                    if pair in matched_pairs:
                        return False, 0
                    matched_pairs.add(pair)
                    count += 1
                else:
                    return False, 0
                    
        if stack:
            return False, 0
        
        if not matched_pairs.issubset({('T1', '/T1'), ('T2', '/T2'), ('T3', '/T3'), ('T4', '/T4'), ('T5', '/T5')}):
            return False, 0
        
        return True, count

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'multihop_qa', 'train_annotations.json')
        data_dict = json.load(open(data_path, "r"))
        for segment in data_dict:
            segment_id = segment['segment_id']
            if segment_id + '.pth.tar' not in self.feature_list:
                continue

            for vqa in segment['annotated_QA']:
                out = {}
                out['id'] = segment_id
                out['duration'] = 180
                out['QA'] = [vqa]

                # TODO: check train data format
                if not self.ensure_format(vqa['T']) or self.find_max_number_in_tokens(vqa['A']) > 5:
                    continue

                paired_tags, num_tags = self.check_and_count_tags(vqa['A'])
                if not paired_tags or num_tags != len(vqa['Evidence'].keys()):
                    continue

                if not all(self.ensure_format(interval) for _, interval in vqa['Evidence'].items()):
                    continue

                self.list_data_dict.append(out)

        print("# Samples: {}".format(len(self.list_data_dict)))
