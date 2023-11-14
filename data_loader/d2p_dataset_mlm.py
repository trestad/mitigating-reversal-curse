from torch.utils.data.dataset import Dataset
import numpy as np
import torch
import os
import random
from transformers import DataCollatorWithPadding, BatchEncoding
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Dict, Optional, Sequence, Union, List
import copy
IGNORE_INDEX = -100
    
class d2pDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.size = len(self.data)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        # return self.data[index]

        inputs = self.data[index]['prompt'] + self.data[index]['completion'] + '</s>'
        inputs = self.tokenizer(inputs, return_tensors="pt")['input_ids'][0]

        return {
            "input_ids": inputs,
        }
