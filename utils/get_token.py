# -*- coding: utf-8 -*-
"""
Created on 2024/9/21 20:20
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/SymTime
"""
from transformers import BertTokenizer, GPT2Tokenizer, DistilBertTokenizer
from typing import Any, Union


def get_tokenizer(
    llm_name: str = "DistilBert",
) -> Union[BertTokenizer, GPT2Tokenizer, DistilBertTokenizer]:
    """
    Get the Tokenizer configuration for large-scale natural language processing

    :param llm_name: The name of the large language model, options include DistilBert, Bert, GPT2

    :return: The corresponding tokenizer object
    """
    if llm_name == "DistilBert":
        try:
            # Try loading from local first
            tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased", trust_remote_code=True, local_files_only=True
            )
        except EnvironmentError:
            # If it does not exist locally, try to download it from the network
            tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased",
                trust_remote_code=True,
                local_files_only=False,
            )
    elif llm_name == "Bert":
        try:
            # Try loading from local first
            tokenizer = BertTokenizer.from_pretrained(
                "google-bert/bert-base-uncased",
                trust_remote_code=True,
                local_files_only=True,
            )
        except EnvironmentError:
            # If it does not exist locally, try to download it from the network
            tokenizer = BertTokenizer.from_pretrained(
                "google-bert/bert-base-uncased",
                trust_remote_code=True,
                local_files_only=False,
            )
    elif llm_name == "GPT2":
        try:
            # Try loading from local first
            tokenizer = GPT2Tokenizer.from_pretrained(
                "openai-community/gpt2", trust_remote_code=True, local_files_only=True
            )
        except EnvironmentError:
            # If it does not exist locally, try to download it from the network
            tokenizer = GPT2Tokenizer.from_pretrained(
                "openai-community/gpt2", trust_remote_code=True, local_files_only=False
            )
    else:
        # Typing error with the name of a large model
        raise ValueError("The llm_name inputs error!")
    return tokenizer
