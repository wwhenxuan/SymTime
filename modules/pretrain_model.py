# -*- coding: utf-8 -*-
"""
Created on 2024/9/30 21:27
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from layers import TSTEncoder
from layers import LLM

from typing import Union, Tuple


class SymTime(nn.Module):
    """SymTime architecture for pre-training"""

    def __init__(self, configs,
                 context_window: int,
                 time_mask_ratio: float,
                 sym_mask_ratio: float) -> None:
        super().__init__()
        self.context_window = context_window
        self.patch_len = configs["patch_len"]
        self.stride = self.patch_len if configs["stride"] is None else configs["stride"]
        self.padding_patch = configs["padding_patch"]
        self.time_layers = configs["time_layers"]
        self.d_model = configs["d_model"]
        self.n_heads = configs["n_heads"]
        self.d_ff = configs["d_ff"]
        self.llm_name = configs["llm_name"]
        self.llm_layers = configs["llm_layers"]
        self.hidden_size = configs["hidden_size"]
        # Freeze the first n layers of parameters of LLM
        self.freeze_layers = configs["freeze_layers"]

        self.embed_dim = configs["embed_dim"]
        # The ratio of momentum model parameter updates
        self.momentum = configs["momentum"]
        # Size of the Momentum Queue
        self.queue_size = configs["queue_size"]
        # Comparative learning annealing parameters
        self.temp = nn.Parameter(torch.ones([]) * configs["temp"])
        # Proportion of false targets in momentum distillation
        self.alpha = configs["alpha"]

        # Mask ratio of the time series and symbol data
        self.time_mask_ratio = time_mask_ratio
        self.sym_mask_ratio = sym_mask_ratio

        # Creating an encoder for time series data
        self.time_encoder = TSTEncoder(patch_len=self.patch_len,
                                       n_layers=self.time_layers,
                                       d_model=self.d_model,
                                       n_heads=self.n_heads,
                                       d_ff=self.d_ff,
                                       norm=configs["norm"],
                                       attn_dropout=configs["attn_dropout"],
                                       dropout=configs["dropout"],
                                       act=configs["act"],
                                       pre_norm=configs["pre_norm"])

        # To obtain time series dimension reduction Token mapping
        self.time_proj = nn.Linear(in_features=self.d_model, out_features=self.embed_dim)

        # Linear mapping for time series patch reconstruction
        self.reconstruct_project = nn.Linear(in_features=self.d_model,
                                             out_features=self.patch_len,
                                             bias=configs["time_project_bias"])

        # Creating an encoder for symbol data
        self.symbolic_encoder = LLM(llm_name=self.llm_name,
                                    llm_layers=self.llm_layers,
                                    hidden_size=configs["hidden_size"],
                                    freeze_layers=self.freeze_layers)

        # To obtain symbol dimension reduction Token mapping
        self.sym_proj = nn.Linear(in_features=self.hidden_size, out_features=self.embed_dim)

        # Get the tokenizer used by the text encoder
        self.tokenizer = self.symbolic_encoder.tokenizer
        # The size of the text capacity
        self.vocab_size = self.tokenizer.vocab_size

        # create momentum models
        self.time_encoder_m = TSTEncoder(patch_len=self.patch_len,
                                         n_layers=self.time_layers,
                                         d_model=self.d_model,
                                         n_heads=self.n_heads,
                                         d_ff=self.d_ff,
                                         norm=configs["norm"],
                                         attn_dropout=configs["attn_dropout"],
                                         dropout=configs["dropout"],
                                         act=configs["act"],
                                         pre_norm=configs["pre_norm"])

        self.time_proj_m = nn.Linear(in_features=self.d_model, out_features=self.embed_dim)
        self.symbolic_encoder_m = LLM(llm_name=self.llm_name,
                                      llm_layers=self.llm_layers,
                                      hidden_size=configs["hidden_size"])
        self.sym_proj_m = nn.Linear(in_features=self.hidden_size, out_features=self.embed_dim)

        # create momentum models params pairs
        self.model_pairs = [[self.time_encoder, self.time_encoder_m],
                            [self.time_proj, self.time_proj_m],
                            [self.symbolic_encoder, self.symbolic_encoder_m],
                            [self.sym_proj, self.sym_proj_m]]
        # copy the params
        self.copy_params()

        # create the queue
        self.register_buffer("time_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("sym_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.time_queue = F.normalize(self.time_queue, dim=0)
        self.sym_queue = F.normalize(self.sym_queue, dim=0)

    def forward(self,
                time: Tensor,
                time_mask: Tensor,
                input_ids: Tensor,
                attn_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        # Masking the time series data
        time_masked, time_attn_mask = self.time_masking(inputs=time, attn_mask=time_mask)
        # time_masked = torch.concat([time_masked, time[:, padding_index:, :]], dim=1)

        # Masking the symbol data as nature language
        labels = input_ids.clone()
        inputs_ids, labels = self.nlp_mask(input_ids=input_ids, vocab_size=self.vocab_size,
                                           device=time.device, targets=labels)

        # Forward propagation of time series data through the time series encoder
        time_embeds = self.time_encoder(x=time_masked, attn_mask=time_attn_mask)
        time_reconstruct = self.reconstruct_project(time_embeds)

        # Get the mask loss_mtm through restruct the missing patch
        loss_mtm = (time_reconstruct[:, 1:, :] - time) ** 2
        loss_mtm = loss_mtm.mean(dim=-1)
        # Only make losses in the masked areas
        loss_mtm = (loss_mtm * (~time_attn_mask).int()).sum() / (~time_attn_mask).sum()

        # Forward propagation for symbolic data in natural language
        sym_outputs = self.symbolic_encoder(inputs_ids, attn_mask, labels)

        # Get the mask loss_mlm through output
        loss_mlm = sym_outputs.loss

        # Get the [CLS] features of time series and symbol data as global features
        time_features = F.normalize(self.time_proj(time_embeds[:, 0, :]), dim=-1)
        sym_features = F.normalize(self.sym_proj(sym_outputs.hidden_states[-1][:, 0, :]), dim=-1)

        # get the momentum features
        with torch.no_grad():
            # Update the parameters of the momentum module
            self.momentum_update()
            time_embeds_m = self.time_encoder_m(x=time_masked, attn_mask=time_attn_mask)
            time_features_m = F.normalize(self.time_proj_m(time_embeds_m[:, 0, :]), dim=-1)
            # time features enqueue
            time_features_all = torch.cat([time_features_m.t(), self.time_queue.clone().detach()], dim=1)

            sym_outputs_m = self.symbolic_encoder_m(inputs_ids, attn_mask, labels)
            sym_features_m = F.normalize(self.sym_proj_m(sym_outputs_m.hidden_states[-1][:, 0, :]), dim=-1)
            # symbol features enqueue
            sym_features_all = torch.cat([sym_features_m.t(), self.sym_queue.clone().detach()], dim=1)
            # Let the time series features match the symbol features [batch_size, batch_size]
            sim_t2s_m = time_features_m @ sym_features_all / self.temp  # s(I, Tm) / tao
            # Let the symbol features match the time series features
            sim_s2t_m = sym_features_m @ time_features_all / self.temp  # s(T, Im) / tao

            sim_targets = torch.zeros(sim_t2s_m.size()).to(time_masked.device)

            sim_targets.fill_diagonal_(1)
            sim_t2s_targets = self.alpha * F.softmax(sim_t2s_m, dim=1) + (1 - self.alpha) * sim_targets
            sim_s2t_targets = self.alpha * F.softmax(sim_s2t_m, dim=1) + (1 - self.alpha) * sim_targets

        sim_t2s = time_features @ sym_features_all / self.temp
        sim_s2t = sym_features @ time_features_all / self.temp

        loss_t2s = -torch.sum(F.log_softmax(sim_t2s, dim=1) * F.softmax(sim_t2s_targets, dim=1), dim=1).mean()
        loss_s2t = -torch.sum(F.log_softmax(sim_s2t, dim=1) * F.softmax(sim_s2t_targets, dim=1), dim=1).mean()

        # let the new features enqueue and the old features dequeue
        self.enqueue_and_dequeue(time_features_m, sym_features_m)

        return loss_mtm, loss_mlm, loss_t2s, loss_s2t

    def time_masking(self,
                     inputs: Tensor,
                     attn_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Function to add mask to time series data"""
        ts = inputs.clone()
        mask = attn_mask.clone()
        # Get batch information
        batch_size, num_tokens, patch_len = inputs.size()
        token_array = torch.sum(attn_mask, dim=1)
        # The proportion of each part that is masked
        num_array = (token_array * self.time_mask_ratio).int()

        for i in range(0, batch_size):
            padding_index = token_array[i]
            number = num_array[i]
            noise = torch.rand(padding_index)
            ids_shuffle = torch.argsort(noise)[: number]
            ts[i, ids_shuffle, :] = 0
            mask[i, ids_shuffle] = False

        return ts, mask

    def nlp_mask(self, input_ids: Tensor,
                 vocab_size: int,
                 device: torch.device,
                 targets: Tensor = None,
                 masked_indices=None) -> Union[Tuple[Tensor, Tensor] or Tensor]:
        """Function to add mask to symbolic data"""
        probability_matrix = torch.full(targets.shape, self.sym_mask_ratio)
        if masked_indices is None:
            # Determine whether the masked content is specified. If not specified, mask it randomly.
            masked_indices = torch.bernoulli(probability_matrix).bool()

        # Make sure the two key tokens pad and cls are not masked out
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        # Used for subsequent loss calculations only on the masked parts
        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    @torch.no_grad()
    def copy_params(self):
        """复制动量模型的参数"""
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize the momentum model params
                param_m.requires_grad = False  # not update the momentum by gradient

    @torch.no_grad()
    def momentum_update(self):
        """更新动量编码器的参数"""
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)

    @torch.no_grad()
    def enqueue_and_dequeue(self, time_features, sym_features):
        """Methods for controlling feature enqueue and dequeue"""
        batch_size = time_features.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.time_queue[:, ptr: ptr + batch_size] = time_features.T
        self.sym_queue[:, ptr: ptr + batch_size] = sym_features.T

        # move the pointer ptr
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
