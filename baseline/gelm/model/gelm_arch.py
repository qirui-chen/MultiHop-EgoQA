# Adapted from https://github.com/NVlabs/LITA. Below is the original copyright:
# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE

import einops
import numpy as np

import torch
import torch.nn as nn

from llava.model.llava_arch import LlavaMetaForCausalLM


class GelmMetaForCausalLM(LlavaMetaForCausalLM):

    def features_to_tokens(self, images):
        assert images.ndim == 3
        tokens = self.encode_features(images)
        return tokens

    def visual_to_tokens(self, images):
        input_type = getattr(self.config, 'input_type', 'image')
        if input_type == 'feature':
            visual_tokens = self.features_to_tokens(images)
            return visual_tokens

    def initialize_time_tokenizer(self, model_args, tokenizer):

        special_tokens = ['<T1>', '</T1>', '<T2>', '</T2>', '<T3>', '</T3>', '<T4>', '</T4>', '<T5>', '</T5>']
        _ = tokenizer.add_tokens(special_tokens)
        self.config.special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)

        saliency_token = ['<T>', '</T>']
        _ = tokenizer.add_tokens(saliency_token)
        self.config.saliency_token_ids = tokenizer.convert_tokens_to_ids(saliency_token)

        self.resize_token_embeddings(len(tokenizer))