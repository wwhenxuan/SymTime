# -*- coding: utf-8 -*-
"""
Created on 2024/9/30 21:30
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
from torch import nn
from torch import Tensor
from transformers import DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer
from typing import Tuple, Any


class LLM(nn.Module):
    """Building LLMs as a general interface for symbolic encoders"""

    def __init__(
        self,
        llm_name: str = "DistilBert",
        llm_layers: int = 6,
        hidden_size: int = 768,
        freeze_layers: int = 3,
    ) -> None:
        super(LLM, self).__init__()
        # Get information about the LLM
        self.llm_name = llm_name
        self.llm_layers = llm_layers
        self.freeze_layers = freeze_layers
        # Get basic config file of LLM
        self.llm_configs, self.llm, self.tokenizer = self.init_llm()
        self.hidden_size = hidden_size
        # Freeze the first n layers of parameters of LLM
        self.freeze()

    def forward(self, input_ids: Tensor, att_mask: Tensor, labels: Tensor) -> Tensor:
        """Forward propagation part of LLM"""
        outputs = self.llm(input_ids, att_mask, labels=labels)
        return outputs  # The loss can be obtained directly from the output of the model

    def freeze(self) -> None:
        """Freeze the first n layers of LLM"""
        for name, param in self.llm.named_parameters():
            for layer_index in range(self.freeze_layers):
                if "layer." + str(layer_index) in name:
                    param.requires_grad = False

    def init_llm(self) -> Tuple[DistilBertConfig, Any, Any]:
        """Select the LLM to use based on the name of the input model"""
        if self.llm_name == "DistilBert":
            llm_config = DistilBertConfig.from_pretrained("distilbert/snapshots/12040accade4e8a0f71eabdb258fecc2e7e948be")
            llm_config.output_attentions = True
            llm_config.output_hidden_states = True
            llm, tokenizer = self.load_DistilBert(config=llm_config)
        else:
            raise ValueError
        return llm_config, llm, tokenizer

    def load_DistilBert(self, config: DistilBertConfig) -> Tuple[Any, Any]:
        """Load DistilBert"""
        try:
            # try to load the pretrained model params from local device
            llm = DistilBertForMaskedLM.from_pretrained(
                "distilbert/snapshots/12040accade4e8a0f71eabdb258fecc2e7e948be",
                # torch_dtype=torch.float16,
                local_files_only=True,
                config=config,
            )
        except EnvironmentError:
            # try to download the pretrained params
            print(
                f"{self.llm_name} not found locally, trying to load from the network..."
            )
            llm = DistilBertForMaskedLM.from_pretrained(
                "distilbert-base-uncased",
                # torch_dtype=torch.float16,
                local_files_only=False,
                config=config,
            )
        try:
            # try to load the tokenizer from local device
            tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert/snapshots/12040accade4e8a0f71eabdb258fecc2e7e948be", local_files_only=True
            )
        except EnvironmentError:
            # try to download the tokenizer
            tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased", local_files_only=False
            )
        return llm, tokenizer
