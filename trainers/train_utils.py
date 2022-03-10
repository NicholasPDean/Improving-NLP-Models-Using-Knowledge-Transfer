import csv
import glob
import json
import random
import logging
import os
from enum import Enum
from typing import List, Optional, Union

import tqdm
import numpy as np

import torch
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
)


def mask_tokens(inputs, tokenizer, args, special_tokens_mask=None):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK,
    10% random, 10% original.
    inputs should be tokenized token ids with size: (batch size X input length).
    """

    # The eventual labels will have the same size of the inputs,
    # with the masked parts the same as the input ids but the rest as
    # args.mlm_ignore_index, so that the cross entropy loss will ignore it.
    labels = inputs.clone()

    # Constructs the special token masks.
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    ##################################################
    # TODO: Please finish this function.

    # First sample a few tokens in each sequence for the MLM, with probability
    # `args.mlm_probability`.
    # Hint: you may find these functions handy: `torch.full`, Tensor's built-in
    # function `masked_fill_`, and `torch.bernoulli`.
    # Check the inputs to the bernoulli function and use other hinted functions
    # to construct such inputs.

    # Function to set a percentage of the indices as True,
    # without including the indices corresponding to special tokens
    def selected_indices(prob, shape=inputs.shape, special_mask=special_tokens_mask):
        prob_tensor = torch.full(shape, prob)
        prob_tensor.masked_fill_(special_mask, 0)
        bern_result = torch.bernoulli(prob_tensor)
        return torch.logical_and(bern_result, bern_result)
    
    indices_masked = selected_indices(args.mlm_probability)
    
    # Remember that the "non-masked" parts should be filled with ignore index.
    ignore_index = args.mlm_ignore_index

    for i in range(len(indices_masked[0])):
        if not indices_masked[0][i]:
            labels[0][i] = ignore_index

    # For 80% of the time, we will replace masked input tokens with  the
    # tokenizer.mask_token (e.g. for BERT it is [MASK] for for RoBERTa it is
    # <mask>, check tokenizer documentation for more details)
    indices_replaced = torch.logical_and(selected_indices(0.8), indices_masked)
    mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    for i in range(len(indices_replaced[0])):
        if indices_replaced[0][i]:
            inputs[0][i] = mask_token_id

    # For 10% of the time, we replace masked input tokens with random word.
    # Hint: you may find function `torch.randint` handy.
    # Hint: make sure that the random word replaced positions are not overlapping
    # with those of the masked positions, i.e. "~indices_replaced".
    indices_randomed = torch.logical_and((torch.logical_and(selected_indices(0.5), ~indices_replaced)), indices_masked)
    rand_ids = torch.randint(low=0, high=tokenizer.vocab_size, size=inputs.shape)

    for i in range(len(indices_randomed[0])):
        if indices_randomed[0][i]:
            inputs[0][i] = rand_ids[0][i]

    # End of TODO
    ##################################################

    # For the rest of the time (10% of the time) we will keep the masked input
    # tokens unchanged
    pass  # Do nothing.

    return inputs, labels


def pairwise_accuracy(guids, preds, labels):

    acc = 0.0  # The accuracy to return.
    
    ########################################################
    # TODO: Please finish the pairwise accuracy computation.
    # Hint: Utilize the `guid` as the `guid` for each
    # statement coming from the same complementary
    # pair is identical. You can simply pair the these
    # predictions and labels w.r.t the `guid`. 
    
    match = True
    prev = 0
    correct = 0
    wrong = 0

    for curr in range(len(guids)):
        if guids[curr] != guids[prev]:
            if match:
                correct += 1
            else:
                wrong += 1
            match = True
        else:
            match = match and (preds[curr] == labels[curr])
        prev = curr
    
    if match:
        correct += 1
    else:
        wrong += 1
    
    acc = correct/(correct+wrong)

    # End of TODO
    ########################################################
     
    return acc


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    pass


if __name__ == "__main__":

    class mlm_args(object):
        def __init__(self):
            self.mlm_probability = 0.4
            self.mlm_ignore_index = -100
            self.device = "cpu"
            self.seed = 42
            self.n_gpu = 0

    args = mlm_args()
    set_seed(args)

    # Unit-testing the MLM function.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    input_sentence = "I am a good student and I love NLP."
    input_ids = tokenizer.encode(input_sentence)
    input_ids = torch.Tensor(input_ids).long().unsqueeze(0)
    
    inputs, labels = mask_tokens(input_ids, tokenizer, args,
                                 special_tokens_mask=None)
    inputs, labels = list(inputs.numpy()[0]), list(labels.numpy()[0])
    ans_inputs = [101, 146, 103, 170, 103, 2377, 103, 146, 1567, 103, 2101, 119, 102]
    ans_labels = [-100, -100, 1821, -100, 1363, -100, 1105, -100, -100, 21239, -100, -100, -100]
    
    if inputs == ans_inputs and labels == ans_labels:
        print("Your `mask_tokens` function is correct!")
    else:
        raise NotImplementedError("Your `mask_tokens` function is INCORRECT!")


    # Unit-testing the pairwise accuracy function.
    guids = [0, 0, 1, 1, 2, 2, 3, 3]
    preds = np.asarray([0, 0, 1, 0, 0, 1, 1, 1])
    labels = np.asarray([1, 0,1, 0, 0, 1, 1, 1])
    acc = pairwise_accuracy(guids, preds, labels)
    
    if acc == 0.75:
        print("Your `pairwise_accuracy` function is correct!")
    else:
        raise NotImplementedError("Your `pairwise_accuracy` function is INCORRECT!")

    ####
