import os
import json
from pathlib import Path
import copy, math
import clip
import torch
import torch.nn as nn
from torch import nn, Tensor
import torch.nn.functional as nnf
from typing import Dict
from collections import OrderedDict
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import GPT2Tokenizer

from .tokenizer import Tokenizer as LLAMATokenizer

import numpy as np

def token_llama2gpt2(labels):

    labels = labels.detach().cpu().numpy()

    llama_tokenizer = LLAMATokenizer(model_path="/data/jinbu/nuscenes-caption/Attribute/LLaMA-Adapter//LLaMA-7B/tokenizer.model")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    raw_sentences = llama_tokenizer.decode(labels)
    tokenized_captions = gpt2_tokenizer.batch_encode_plus(raw_sentences)['input_ids']
    


    return torch.tensor(tokenized_captions)

def position_embedding(max_len: int, d_model: int) -> Tensor:
    position_embedding = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         -(math.log(10000.0) / d_model))
    position_embedding[:, 0::2] = torch.sin(position * div_term)
    position_embedding[:, 1::2] = torch.cos(position * div_term)
    return position_embedding

class GPT2Captioner(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding_size = 256
        self.max_positions = 64
        self.max_des_len = 64
        
        ## initialize tokenizer for batch decoding
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.nvocabs = len(self.tokenizer)
        
        ## caption generation cores
        gpt2_config = GPT2Config(
            vocab_size=self.nvocabs,
            n_positions=self.max_positions,
            n_embd=self.embedding_size,
            n_layer=2,
            n_head=4,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            add_cross_attention=True,
        )
        self.transformer = GPT2LMHeadModel(config=gpt2_config)
        self.transformer.transformer.wpe = nn.Embedding.from_pretrained(
            position_embedding(self.max_positions, self.embedding_size)
        )
        
        ## for proposal feature projection
        self.feature_projector = nn.Sequential(
            nn.Linear(256, self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
        )
        
        self.context_projector = nn.Sequential(
            nn.Linear(256, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
        )
        
        ## ---- super parameters for evaluation
        self.caption_config = {
            'early_stopping': True,
            'eos_token_id': self.tokenizer.eos_token_id,
            'num_beams': 1
        }


    def get_trainable_params(self, phase='finetune'):
        for name, para in self.named_parameters():
            para.requires_grad = False

        if phase == 'finetune':
            for name, para in self.named_parameters():
                if name.startswith("llama."):
                    if 'norm' in name or 'bias' in name:
                        para.data = para.data.float()
                        para.requires_grad = True

        elif phase == 'pretrain':
            train_param_name = ['gate', 'bev_proj', 'bev_proj_norm', 'bbox_query', 'visual_blocks', 'visual_proj', 'visual_proj_norm', 'adapter_query']
            for name, para in self.named_parameters():
                for train_name in train_param_name:
                    if train_name in name:
                        para.data = para.data.float()
                        para.requires_grad = True
        
        else:
            raise ValueError(f"Unknown model phase: {phase}")


    def forward_visual(self, det_inputs):
        feats, pred = det_inputs

        feats = feats.permute(0, 2, 1) # B 256 2500
        
        bev_size = int(feats.size(-1)**0.5)
        
        feats = feats.contiguous().view(feats.size(0), self.bev_dim, 1, bev_size, bev_size)
        feats = self.downsample(feats)
        # feats = feats.contiguous().view(len(feats), self.bev_dim, -1)
        feats = feats.view(len(feats), self.bev_dim, -1)
        feats = feats.permute(0, 2, 1)    # B bev_size/5*bev_size/5 256

        clip_feats = self.bev_proj_norm(self.bev_proj(feats.float()))

        bbox_query = self.bbox_query(pred.unsqueeze(-2))
        bbox_query = bbox_query    # B 1 768

        bbox_query = torch.cat([bbox_query, clip_feats], dim=1)
        for block in self.visual_blocks:
            bbox_query = block(bbox_query)

        bbox_query = bbox_query[:, :self.query_len, :]
        bbox_query = self.visual_proj(bbox_query)
        bbox_query = self.visual_proj_norm(bbox_query)

        return bbox_query

    def forward(self, cap_inputs, det_inputs):
        tokens, labels, c_weights = cap_inputs
        feats, obj_pred_box = det_inputs

        gt_box_cap_label = token_llama2gpt2(labels)

        bbox_query = self.forward_visual(det_inputs)

        inputs_embeds = torch.cat([
            bbox_query, self.transformer.transformer.wte(gt_box_cap_label)
        ], dim=2)   # batch x nproposals x (nprefix + max_des_len) x channel

        outputs = self.transformer( # num_annotated x (1 + max_des_len)
            inputs_embeds=inputs_embeds[annotated_proposal == 1],
            attention_mask=inputs_masks[annotated_proposal == 1],
            encoder_hidden_states=\
                None if detector_output.get('encoder_hidden_states', None) is None else \
                    detector_output['encoder_hidden_states'][annotated_proposal == 1]
        )


        h = self.llama.norm(h)
        output = self.llama.output(h)
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            assert self.llama.vocab_size == 32000
            c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())
            # c_weights = c_weights.unsqueeze(1).repeat(1, output.shape[1]).view(-1)
            # c_loss = (c_loss*c_weights).mean()
            if c_loss < 0:
                print(f"c_loss is {c_loss}")

        return c_loss

    @torch.inference_mode()
    def forward_inference(self, bbox_query, tokens, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, start_pos, freqs_cis, mask)

        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers[-1 * self.query_layer:]:
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            dynamic_adapter = dynamic_adapter + bbox_query
            h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
            adapter_index = adapter_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float()

