# Adapted from https://github.com/NVlabs/LITA. Below is the original copyright:
# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE

import numpy as np
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import transformers

from gelm.arguments import DataArguments
from gelm.constants import IGNORE_INDEX
from gelm.dataset.temporal_reasoning_dataset import TemporalReasoningDataset_activitynet
from gelm.dataset.multihop_qa_dataset import MultiHopQADataset_ego4d


class HybridDataset(Dataset):
    def __init__(self, data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(HybridDataset, self).__init__()


        self.tasks = data_args.tasks.split("||")  # TODO: change to tasks?
        task_sample_rate = data_args.task_sample_rate
        s = sum(task_sample_rate)
        self.task_sample_rate = [float(x)/s for x in task_sample_rate]
        assert len(self.task_sample_rate) == len(self.tasks)

        ds_dict = {
            'temporal_reasoning':{
                'activitynet': TemporalReasoningDataset_activitynet,
            },
            'multihop_qa':{
                'ego4d': MultiHopQADataset_ego4d,
            }
        }

        self.all_datasets = []
        self.all_sample_rate = []
        for task in self.tasks:
            task_data = getattr(data_args, task + '_data', '')
            datasets = []
            sample_counts = []
            for data in task_data.split('||'):
                dataset = ds_dict[task][data](data_path, tokenizer, data_args)
                datasets.append(dataset)
                sample_counts.append(len(dataset))   
            sample_rate = getattr(data_args, task + '_sample_rate', sample_counts)
            assert len(sample_rate) == len(datasets)
            s = sum(sample_rate)
            sample_rate = [float(x)/s for x in sample_rate]
            self.all_sample_rate.append(sample_rate)
            self.all_datasets.append(datasets)

    def __len__(self):
        samples = 0
        for task in self.all_datasets:
            for dataset in task:
                samples += len(dataset)
        # print(samples)
        return samples

    def __getitem__(self, idx):
        rng = np.random.RandomState()  # local rng independent of global
        task = rng.choice(list(range(len(self.all_datasets))), p=self.task_sample_rate)
        dataset = rng.choice(list(range(len(self.all_datasets[task]))), p=self.all_sample_rate[task])
        return self.all_datasets[task][dataset][0]  # random sample idx inside so just input 0


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images


        if all('saliency' in instance for instance in instances):
            batch['saliency'] = torch.stack([instance['saliency'] for instance in instances])
        if all('gt_spans' in instance for instance in instances):
            batch['gt_spans'] = torch.stack([instance['gt_spans'] for instance in instances])
        if all('evidence' in instance for instance in instances):
            batch['evidence'] = pad_sequence(
            [instance['evidence'] for instance in instances], batch_first=True, padding_value=0)
        return batch