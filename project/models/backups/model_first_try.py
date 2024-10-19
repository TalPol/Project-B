import numpy as np
from bonito.nn import Permute, layers
import torch
from torch.nn.functional import log_softmax, ctc_loss
from torch.nn import Module, ModuleList, Sequential, Conv1d, BatchNorm1d, Dropout

from fast_ctc_decode import beam_search, viterbi_search

import mamba_ssm
from mamba_ssm import Mamba2

class Model(Module):

    def __init__(self, config):
        super(Model, self).__init__()
        if 'qscore' not in config:
            self.qbias = 0.0
            self.qscale = 1.0
        else:
            self.qbias = config['qscore']['bias']
            self.qscale = config['qscore']['scale']
        self.config = config
        self.dim = config['block'][0]['dim']
        self.hdim = config['block'][0]['headdim']
        self.state = config['block'][0]['state']
        self.kernel = config['block'][0]['conv']
        self.expand = config['block'][0]['expand']
        self.alphabet = config['labels']['labels']

        self.mamba = Mamba2(d_model=self.dim, d_state=self.state, d_conv=self.kernel, expand=self.expand, headdim=self.hdim)

    def forward(self, x):
        return self.mamba(x)
    
    def decode(self, x, beamsize=5, threshold=1e-3, qscores=False, return_path=False):
        x = x.exp().numpy().astype(np.float32)
        if beamsize == 1 or qscores:
            seq, path  = viterbi_search(x, self.alphabet, qscores, self.qscale, self.qbias)
        else:
            seq, path = beam_search(x, self.alphabet, beamsize, threshold)
        if return_path: return seq, path
        return seq
    
    def ctc_label_smoothing_loss(self, log_probs, targets, lengths, weights=None):
        T, N, C = log_probs.shape
        weights = weights or torch.cat([torch.tensor([0.4]), (0.1 / (C - 1)) * torch.ones(C - 1)])
        log_probs_lengths = torch.full(size=(N, ), fill_value=T, dtype=torch.int64)
        if len(lengths) != N or log_probs_lengths.shape[0] != N:
            raise ValueError(f"Lengths mismatch: expected batch size {N}, but got lengths {len(lengths)} and log_probs_lengths {log_probs_lengths.shape[0]}")
        
        loss = ctc_loss(log_probs.to(torch.float32), targets, log_probs_lengths, lengths, reduction='mean')
        label_smoothing_loss = -((log_probs * weights.to(log_probs.device)).mean())
        return {'total_loss': loss + label_smoothing_loss, 'loss': loss, 'label_smooth_loss': label_smoothing_loss}

    def loss(self, log_probs, targets, lengths):
        return self.ctc_label_smoothing_loss(log_probs, targets, lengths)
