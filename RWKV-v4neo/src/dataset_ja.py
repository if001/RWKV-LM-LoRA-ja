########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime
from .utils import TOKENIZER


class DatasetJA(Dataset):
    def __init__(self, args):
        self.args = args        
        self.ctx_len = args.ctx_len
        self.data = open(args.data_file, "r", encoding=args.data_type).read()
        self.data_size = len(self.data)

        self.t = TOKENIZER([
                "20B_tokenizer.json",
                "20B_tokenizer.json",
        ])
        self.data = open(args.data_file, "r", encoding=args.data_type).read()
        self.vocab_size = self.t.vocab_size

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def _encode(self, d):
       return self.t.tokenizer.encode(
            d,
            padding="max_length",
            max_length=self.ctx_len,
            return_tensors="pt",
            add_special_tokens=False,
            dtype=torch.long
       )
    def __getitem__(self, idx):
        data = self.data[idx: self.ctx_len]
        x = self._encode(data[1])
        y = self._encode(data[:-1])
        return x, y
