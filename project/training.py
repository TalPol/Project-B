import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import optuna

import mamba_ssm
from mamba_ssm import Mamba2

from glob import glob
from functools import partial
from time import perf_counter
from collections import OrderedDict
from datetime import datetime
import re

from bonito.schedule import linear_warmup_cosine_decay
from bonito.util import accuracy, decode_ref, permute, concat, match_names, tqdm_environ
import bonito

import pod5
from pod5 import Reader
from Bio import SeqIO

from tqdm import tqdm
import torch.cuda.amp as amp

def load_state(dirname, device, model, optim=None):
    """
    Load a model state dict from disk
    """
    model.to(device)
    if hasattr(model, "module"):
        model = model.module

    weight_no = optim_no = None

    optim_files = glob(os.path.join(dirname, "optim_*.tar"))
    optim_nos = {int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in optim_files}

    weight_files = glob(os.path.join(dirname, "weights_*.tar"))
    weight_nos = {int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files}

    if optim is not None:
        weight_no = optim_no = max(optim_nos & weight_nos, default=None)
    else:
        weight_no = max(weight_nos, default=None)

    to_load = []
    if weight_no:
        to_load.append(("weights", model))
    if optim_no:
        to_load.append(("optim", optim))

    if to_load:
        print("[picking up %s state from epoch %s]" % (', '.join([n for n, _ in to_load]), weight_no))
        for name, obj in to_load:
            state_dict = torch.load(
                os.path.join(dirname, '%s_%s.tar' % (name, weight_no)), map_location=device
            )
            if name == "weights":
                state_dict = {k2: state_dict[k1] for k1, k2 in match_names(state_dict, obj).items()}
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                state_dict = new_state_dict
            obj.load_state_dict(state_dict)
        epoch = weight_no
    else:
        epoch = 0

    return epoch

def load_translations(fasta_files):
    translations = {}
    for fasta_file in fasta_files:
        for record in SeqIO.parse(fasta_file, "fasta"):
            translations[record.id] = str(record.seq)
    return translations

class Pod5BasecallerDataset(Dataset):
    def __init__(self, pod5_files, fasta_files, chunk_size=2000, transform=None):
        self.pod5_files = pod5_files
        self.fasta_files = fasta_files
        self.translation = load_translations(fasta_files)
        self.chunk_size = chunk_size
        self.transform = transform
        

    def __len__(self):
        # Total number of chunks in all files
        total_chunks = 0
        for file in self.pod5_files:
            with Reader(file) as reader:
                data_len = 0
                for read in read.reads():
                    data_len += len(read.signal)  # Assume this gives total sequence length
                
                total_chunks += (data_len + self.chunk_size - 1) // self.chunk_size
        return total_chunks

    def __getitem__(self, idx):
        # Calculate which file and chunk within the file corresponds to `idx`
        file_idx, chunk_idx = self._get_file_and_chunk_idx(idx)
        
        with Reader(self.pod5_files[file_idx]) as reader:
            for read in reader.reads():
                data = read.signal
            read = next()
            data = pod5.get_sequence_data()
            label = pod5.get_label_data()

            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, len(data))
            
            chunk_data = data[start:end]
            chunk_label = label[start:end]

            if self.transform:
                chunk_data = self.transform(chunk_data)

            return torch.tensor(chunk_data, dtype=torch.float32), torch.tensor(chunk_label, dtype=torch.long)
    
    def _get_file_and_chunk_idx(self, idx):
        # Map global idx to file index and chunk index within the file
        accumulated_chunks = 0
        for file_idx, file in enumerate(self.pod5_files):
            with Reader(file) as pod5:
                data_len = len(pod5.get_sequence_data())
                num_chunks = (data_len + self.chunk_size - 1) // self.chunk_size
                
                if accumulated_chunks + num_chunks > idx:
                    return file_idx, idx - accumulated_chunks
                accumulated_chunks += num_chunks
        raise IndexError("Index out of range")

# Custom collate function for variable-length chunks
def collate_fn(batch):
    data, labels = zip(*batch)
    data_padded = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
    return data_padded, labels_padded


class Trainer:
    def __init__(
        self, model, device, train_loader, valid_loader, criterion=None,
        use_amp=True, lr_scheduler_fn=None, restore_optim=False,
        save_optim_every=10
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion or model.loss
        self.use_amp = use_amp
        self.lr_scheduler_fn = lr_scheduler_fn or linear_warmup_cosine_decay()
        self.restore_optim = restore_optim
        self.save_optim_every = save_optim_every
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.optimizer = None
