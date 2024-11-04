import os
import pickle as pkl
import numpy as np 

from typing import List, Tuple
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import Dataset

from build_data import build_data_tokenized

def pad_var_sequences(padded_sequences: torch.Tensor, lengths): # padded_sequences: (B, L, H)
    # note: the batch is a sorted list with descending length
    packed_sequences = rnn_utils.pack_padded_sequence(padded_sequences, lengths, batch_first=True)
    return packed_sequences

def pack_var_sequences(packed_sequences, padding_value = 0., total_length = None): # return : (B, L, Max_Len)
    padded_sequences, _ = rnn_utils.pad_packed_sequence(packed_sequences, batch_first=True, padding_value=padding_value, total_length=total_length)
    return padded_sequences

def sequence_mask(lengths, max_length = None, device = "cpu"): # mask 1 for padding, 0 for non-padding
    lengths = torch.tensor(lengths, dtype=torch.int64)
    if max_length is None:
        max_length = lengths.max()
    mask = torch.arange(max_length).expand(len(lengths), max_length) >= lengths.unsqueeze(1) # length is on cpu
    return mask.to(device)

def sequence_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], max_length: int = 100, device = "cpu"):
    sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    # padding src tensor
    src, trg = zip(*sorted_batch)
    src_len = [len(seq) for seq in src]
    src = rnn_utils.pad_sequence(src, batch_first=True).to(device)
    # padding trg tensor
    trg = list(trg)
    trg_len = [len(seq) for seq in trg]
    for i in range(len(trg)):
        trg[i] = F.pad(trg[i], (0, max_length - len(trg[i])), value=0).to(device)
    trg = torch.stack(trg, dim=0) # (batch_size, max_length) batch_first = True
    return (src, src_len), (trg, trg_len)

import pickle as pkl
from torch.utils.data import Dataset
import torch
import os
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, data_dir: str, vocab_en: dict, vocab_zh: dict, max_length: int = 100, block_ids=1):
        self.data_dir = data_dir
        self.max_length = max_length
        self.block_ids = block_ids
        self.vocab_en = vocab_en  # 英文词汇表：词 -> ID
        self.vocab_zh = vocab_zh  # 中文词汇表：词 -> ID
        self.process(block_ids)

    def __len__(self):
        return len(self.en_sequences)

    def __getitem__(self, idx):
        if not isinstance(idx, (int, np.integer)):
            raise TypeError(f'Index must be an integer, got {type(idx).__name__}')
        
        idx = int(idx)
        
        if idx < 0 or idx >= len(self.en_sequences):
            raise IndexError(f'Index {idx} out of bounds')

        en_seq = self.en_sequences[idx]
        zh_seq = self.zh_sequences[idx]

        if isinstance(en_seq, dict):
            en_seq = list(en_seq.values())
        if isinstance(zh_seq, dict):
            zh_seq = list(zh_seq.values())

        en_seq = en_seq[:self.max_length]
        zh_seq = zh_seq[:self.max_length]

        if not all(isinstance(token, int) for token in en_seq):
            raise ValueError(f"EN sequence at index {idx} contains non-integer tokens: {en_seq}")
        if not all(isinstance(token, int) for token in zh_seq):
            raise ValueError(f"ZH sequence at index {idx} contains non-integer tokens: {zh_seq}")

        try:
            return torch.tensor(en_seq, dtype=torch.int64), torch.tensor(zh_seq, dtype=torch.int64)
        except Exception as e:
            print(f"Error converting sequence at index {idx}: {e}")
            print(f"EN sequence: {en_seq}")
            print(f"ZH sequence: {zh_seq}")
            raise

    def process(self, block_ids=1):
        self.block_ids = block_ids
        print(f"Load SequenceDataset at {self.data_dir}")
        
        en_file = f'{self.data_dir}/sentences_en_ids_{block_ids}.pkl'
        zh_file = f'{self.data_dir}/sentences_zh_ids_{block_ids}.pkl'

        if not os.path.exists(en_file) or not os.path.exists(zh_file):
            raise FileNotFoundError(f"Data files not found. Please ensure {en_file} and {zh_file} exist.")

        print("Loading the data...")
        with open(en_file, 'rb') as f:
            self.en_sequences = pkl.load(f)
        with open(zh_file, 'rb') as f:
            self.zh_sequences = pkl.load(f)

        print(f"Type of en_sequences: {type(self.en_sequences)}")
        print(f"Type of zh_sequences: {type(self.zh_sequences)}")
        if len(self.en_sequences) > 0:
            print(f"Type of first en item: {type(self.en_sequences[0])}")
            print(f"First EN sequence: {self.en_sequences[0]}")
            print(f"Type of first zh item: {type(self.zh_sequences[0])}")
            print(f"First ZH sequence: {self.zh_sequences[0]}")

        for i, seq in enumerate(self.en_sequences):
            if not all(isinstance(token, int) for token in seq):
                print(f"EN sequence at index {i} contains non-integer tokens: {seq}")

        for i, seq in enumerate(self.zh_sequences):
            if not all(isinstance(token, int) for token in seq):
                print(f"ZH sequence at index {i} contains non-integer tokens: {seq}")

        if not isinstance(self.en_sequences, list) or not isinstance(self.zh_sequences, list):
            raise TypeError("Loaded data must be a list of sequences")

        if len(self.en_sequences) != len(self.zh_sequences):
            raise ValueError("Mismatch in number of English and Chinese sequences")

        print(f"Loaded {len(self.en_sequences)} sequence pairs")
        print("Done")

            