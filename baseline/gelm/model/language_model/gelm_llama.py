import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from gelm.model.gelm_arch import GelmMetaForCausalLM
from gelm.model.grounding_head import (Conv, LinearLayer,
                                       generalized_temporal_iou,
                                       merge_intervals, temporal_nms)
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.model.language_model.llava_llama import LlavaLlamaModel
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoConfig, AutoModelForCausalLM, LlamaConfig,
                          LlamaForCausalLM, LlamaModel)
from transformers.modeling_outputs import CausalLMOutputWithPast


class GelmConfig(LlamaConfig):
    model_type = "gelm"


class GelmLlamaForCausalLM(LlamaForCausalLM, GelmMetaForCausalLM):
    config_class = GelmConfig

    def __init__(self, config, **kwargs):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if 'grounding_args' in kwargs:
            # Training
            grounding_args = kwargs.get("grounding_args", None)
        else:
            # Inference
            grounding_args = config.grounding_args
        # print(grounding_args)
        self.grounding_module = GroundingModule(config.hidden_size,
                                                d_model=grounding_args.get("d_gnd", None),
                                                d_proj=grounding_args.get("d_proj", None),
                                                n_layers=grounding_args.get("gnd_enc_layers", None)).to(torch.bfloat16)

        # Initialize weights and apply final processing
        self.post_init()

        self.lm_loss = 0
        self.bce_loss = 0
        self.nce_loss = 0

    def get_additional_loss(self):
        return {
            'lm_loss': self.lm_loss,
            'bce_loss': self.bce_loss,
            'nce_loss': self.nce_loss,
        }

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        saliency: Optional[torch.Tensor] = None,
        evidence: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        before_input_ids = input_ids

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,  # None
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            self.lm_loss = loss.item()

        if evidence is not None and saliency is not None:
            v_len = images.shape[1]
            # Enable model/pipeline parallelism
            device = inputs_embeds.device
            evidence = evidence.to(device)  # (B, #max_query, T)
            saliency = saliency.to(device)  # (B, T)

            image_embeds, batched_queries, num_queries = [], [], []
            for batch_idx, cur_input_ids in enumerate(before_input_ids):
                image_index = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                image_embeds.append(hidden_states[batch_idx, image_index:image_index + v_len])  # (T, D)

                query_indices = self.find_query_indices(cur_input_ids)
                queries = []

                for start, end in query_indices:
                    start_embed = hidden_states[batch_idx, start + (v_len - 1) - 1]  # hidden states of <Ti> - 1 (D)
                    end_embed = hidden_states[batch_idx, end + (v_len - 1) - 1]  # hidden states of </Ti> - 1 (D)
                    merged_query_embed = torch.cat([start_embed, end_embed], dim=-1)
                    queries.append(merged_query_embed)  # (1, 2D)

                batched_queries.append(torch.stack(queries, dim=0))  # (#query, 2D)
                num_queries.append(batched_queries[-1].shape[0])

            image_embeds = torch.stack(image_embeds)  # (B, T, D)
            batched_queries = pad_sequence(batched_queries, batch_first=True, padding_value=0)  # (B, #max_query, 2D)
            queries_padding_mask = pad_sequence([torch.zeros(num) for num in num_queries],
                                                batch_first=True,
                                                padding_value=1).to(device)  # (B, #max_query)

            pred_saliency, image_proj, query_proj = self.grounding_module(image_embeds, batched_queries,
                                                                          queries_padding_mask)
            loss_nce = contrastive_loss(image_proj, query_proj, queries_padding_mask.bool(), evidence.bool())
            self.nce_loss = loss_nce.item()
            loss += loss_nce

            weights = torch.ones_like(saliency, dtype=torch.bfloat16)
            weights[~saliency.bool()] = .1
            loss_bce = F.binary_cross_entropy(pred_saliency.squeeze(-1), saliency, weight=weights, reduction="mean")
            loss_bce *= 10.0
            self.bce_loss = loss_bce.item()
            loss += loss_bce

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def find_query_indices(self, input_ids):
        query_indices = []
        for i in range(0, len(self.config.special_token_ids), 2):
            first = self.config.special_token_ids[i]
            second = self.config.special_token_ids[i + 1]

            start_indices = (input_ids == first).nonzero(as_tuple=True)
            end_indices = (input_ids == second).nonzero(as_tuple=True)

            if len(start_indices[0]) == 1 and len(end_indices[0]) == 1:
                start_index = start_indices[0].item()
                end_index = end_indices[0].item()
                query_indices.append([start_index, end_index])
            elif (len(start_indices[0]) > 1 or len(end_indices[0]) > 1) and len(start_indices[0]) and len(
                    end_indices[0]):
                # unexpected case <T2></T1> in inference
                start_index = start_indices[0][0].item() if len(start_indices[0]) > 1 else start_indices[0].item()
                end_index = end_indices[0][-1].item() if len(end_indices[0]) > 1 else end_indices[0].item()
                query_indices.append([start_index, end_index])
        if not len(query_indices):
            query_indices = [[0, len(input_ids) - 1]]
        return query_indices

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      attention_mask=None,
                                      inputs_embeds=None,
                                      **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "images": kwargs.get("images", None),
        })
        return model_inputs

    @torch.no_grad()
    def generate_with_time(self, input_ids, **model_inputs):

        multi_hop = model_inputs.pop("multi_hop")
        factor_saliency = model_inputs.pop("factor_saliency")
        factor_similarity = model_inputs.pop("factor_similarity")

        outputs = self.generate(input_ids, output_hidden_states=True, return_dict_in_generate=True,
                                **model_inputs)  # old infer with dict
        output_ids = outputs.sequences  # input + output
        added_ids = output_ids[:, input_ids.shape[1]:]
        output_hidden_states = outputs.hidden_states  # (#added (#layers [B, ?, D])) ?: [0][-1]->input seq (or) [1:][-1]-> 1

        images = model_inputs.get("images", None)  # we have
        image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0]
        image_embeds = output_hidden_states[0][-1][:, image_index:image_index + images.shape[1], :]

        batched_queries = []
        for batch_idx, cur_added_ids in enumerate(added_ids):

            query_indices = self.find_query_indices(cur_added_ids)
            queries = []
            for start, end in query_indices:
                start_embed = output_hidden_states[start][-1][batch_idx, 0].unsqueeze(0)  # (1, D)
                end_embed = output_hidden_states[end][-1][batch_idx, 0].unsqueeze(0)  # (1, D)
                merged_query_embed = torch.cat([start_embed, end_embed], dim=-1)
                queries.append(merged_query_embed)  # (1, 2D)

        batched_queries = torch.cat(queries).unsqueeze(0)  #(B=1, S, 2D)
        queries_padding_mask = torch.zeros(batched_queries.shape[0],
                                           batched_queries.shape[1]).to(batched_queries.device)

        pred_saliency, image_proj, query_proj = self.grounding_module(image_embeds, batched_queries,
                                                                      queries_padding_mask)
        saliency_prob = pred_saliency.squeeze(-1)  # (B, T)

        assert saliency_prob.shape[0] == 1

        saliency_prob = saliency_prob.squeeze()
        saliency_proposals = threshold_prob_proposal(saliency_prob, factor=factor_saliency, multi_proposals=multi_hop)
        saliency_proposals = merge_intervals(saliency_proposals)

        normalized_image_proj = F.normalize(image_proj, dim=-1)  # (bsz, #frames, D)
        normalized_query_proj = F.normalize(query_proj, dim=-1)  # (bsz, #queries, D)
        logits = torch.einsum("BTD,BSD->BST", normalized_image_proj,
                              normalized_query_proj) / 0.07  # (bsz, #queries, #frames)
        similarity_prob = logits.softmax(-1).mean(1)
        similarity_prob = nn.AvgPool1d(3, stride=1, padding=1)(similarity_prob).squeeze()  # (#frames)
        similarity_proposals = threshold_prob_proposal(similarity_prob,
                                                       factor=factor_similarity,
                                                       multi_proposals=multi_hop)
        similarity_proposals = merge_intervals(similarity_proposals)

        return output_ids, saliency_proposals, similarity_proposals


class GroundingModule(nn.Module):

    def __init__(self, lm_hidden_size, d_model=1024, d_proj=128, n_layers=3):
        super(GroundingModule, self).__init__()

        self.image_mlp = nn.Sequential(
            *[nn.ReLU(), LinearLayer(lm_hidden_size, d_model, layer_norm=True, dropout=0.0, relu=False)])
        self.query_mlp = nn.Sequential(
            *[LinearLayer(lm_hidden_size * 2, d_model, layer_norm=True, dropout=0.0, relu=True)])

        self.pos_encoder = nn.Embedding(500, d_model)
        self.token_type_embeddings = nn.Embedding(2, d_model)
        self.encoder = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=d_model, nhead=16, batch_first=True) for i in range(n_layers)])

        self.v_sim_proj = LinearLayer(d_model, d_proj, layer_norm=True, dropout=0.0, relu=True)
        self.q_sim_proj = LinearLayer(d_model, d_proj, layer_norm=True, dropout=0.0, relu=True)

        self.saliency_mlp = Conv(d_model, 128, 1, num_layers=2, kernel_size=3)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights. This method should be overridden by derived class.
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, val=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=.02)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, val=1.0)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, image_embed, batched_queries, queries_padding_mask):
        """
        image_embed: (B, T, D)
        batched_queries: (B, S=#max_queries, D)
        padding_mask: (B, S=#max_queries)
        """

        image_embed = self.image_mlp(image_embed) + self.token_type_embeddings.weight[0]
        query_embed = self.query_mlp(batched_queries) + self.token_type_embeddings.weight[1]

        image_proj = self.v_sim_proj(image_embed)
        query_proj = self.q_sim_proj(query_embed)

        tokens = torch.cat([image_embed, query_embed], dim=1)  # (B, S+T, D)
        tokens = tokens + self.pos_encoder.weight[None, :tokens.shape[1], :]
        padding_mask = torch.cat([
            torch.zeros(image_embed.shape[0], image_embed.shape[1], device=queries_padding_mask.device),
            queries_padding_mask
        ],
                                 dim=1).bool()  # (B, S+T)

        for layer in self.encoder:
            tokens = layer(tokens, src_key_padding_mask=padding_mask)

        image_tokens = tokens[:, :image_embed.shape[1], :]
        query_tokens = tokens[:, -query_embed.shape[1]:, :]
        pred_saliency = self.saliency_mlp(image_tokens).sigmoid()

        return pred_saliency, image_proj, query_proj


def contrastive_loss(image_proj, query_proj, queries_padding_mask, evidence):
    """
    queries_padding_mask: (B, S=#max_queries)
    evidence: (B, S, T)
    """
    normalized_image_proj = F.normalize(image_proj, dim=-1)  # (bsz, #frames, D)
    normalized_query_proj = F.normalize(query_proj, dim=-1)  # (bsz, #queries, D)
    logits = torch.einsum("BTD,BSD->BST", normalized_image_proj, normalized_query_proj)  # (bsz, #queries, #frames)

    logits = logits / 0.07  # (bsz, #queries, #frames)

    pos_logits = logits.masked_fill(~evidence, -6e4)
    pos_neg_logits = logits.masked_fill(queries_padding_mask.unsqueeze(-1).repeat(1, 1, logits.shape[-1]), -6e4)

    # sum similarity along T
    pos_term = pos_logits.logsumexp(2)  # (bsz, #queries)
    neg_term = pos_neg_logits.logsumexp(2)  # (bsz, #queries)

    loss_nce = -pos_term + neg_term  # (bsz, #queries)
    loss_nce = loss_nce.masked_fill(queries_padding_mask, 0).mean()

    return loss_nce


def threshold_prob_proposal(prob, factor=0.7, multi_proposals=True):

    max_idx = torch.argmax(prob).item()
    max_value = prob[max_idx].item()
    threshold = factor * max_value

    if multi_proposals:
        proposals = []
        start = None
        for idx, value in enumerate(prob):
            if value > threshold:
                if start is None:
                    start = idx
            else:
                if start is not None:
                    end = idx - 1
                    proposals.append([start, end])
                    start = None
        if start is not None:
            proposals.append([start, len(prob) - 1])
    else:
        start = max_idx
        while start > 0 and prob[start] > threshold:
            start -= 1
        end = max_idx
        while end < prob.shape[-1] - 1 and prob[end] > threshold:
            end += 1
        proposals = [start, end]

    return proposals


AutoConfig.register("gelm", GelmConfig)
AutoModelForCausalLM.register(GelmConfig, GelmLlamaForCausalLM)
