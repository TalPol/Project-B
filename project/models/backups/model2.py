import logging
import types
from functools import lru_cache

logger = logging.getLogger(__name__)
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential, Conv1d, BatchNorm1d, Dropout
try:
    from flash_attn import flash_attn_qkvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn.modules.mlp import GatedMlp
    from flash_attn.ops.triton.layer_norm import RMSNorm
except ImportError:
    logger.warning(
        "please install flash-attn to use the transformer module: "
        "`pip install flash-attn --no-build-isolation`"
    )
from torch.nn.functional import log_softmax, ctc_loss
from fast_ctc_decode import beam_search, viterbi_search

from bonito.crf.model import SeqdistModel
from bonito.nn import from_dict, register, LinearCRFEncoder, MakeContiguous, Module, Permute, Serial

import mamba_ssm
from mamba_ssm import Mamba2

def deepnorm_params(depth):
    """
    Returns the DeepNorm (https://arxiv.org/abs/2203.00555) alpha and beta parameters.
    """
    alpha = round((2*depth)**0.25, 7)
    beta = round((8*depth)**(-1/4), 7)
    return alpha, beta


@lru_cache(maxsize=2)
def sliding_window_mask(seq_len, window, device):
    band = torch.full((seq_len, seq_len), fill_value=1.0)
    band = torch.triu(band, diagonal=-window[0])
    band = band * torch.tril(band, diagonal=window[1])
    band = band.to(torch.bool).to(device)
    return band

'''
class MultiHeadAttention(Module):
    def __init__(self, d_model, nhead, qkv_bias=False, out_bias=True, rotary_dim=None, attn_window=None):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_dim = self.head_dim if rotary_dim is None else rotary_dim

        self.Wqkv = torch.nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=out_bias)

        self.rotary_emb = RotaryEmbedding(self.rotary_dim, interleaved=False)
        self.attn_window = (-1, -1) if attn_window is None else tuple(attn_window)

    def attn_func(self, qkv):
        if torch.cuda.get_device_capability(qkv.device)[0] >= 8 and (torch.is_autocast_enabled() or qkv.dtype == torch.half):
            attn_output = flash_attn_qkvpacked_func(qkv, window_size=self.attn_window)
        else:
            q, k, v = torch.chunk(qkv.permute(0, 2, 3, 1, 4), chunks=3, dim=1)
            mask = sliding_window_mask(qkv.shape[1], self.attn_window, q.device)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            attn_output = attn_output.permute(0, 1, 3, 2, 4)
        return attn_output

    def forward(self, x):
        N, T, _ = x.shape

        qkv = self.Wqkv(x).view(N, T, 3, self.nhead, self.head_dim)

        qkv = self.rotary_emb(qkv)

        attn_output = self.attn_func(qkv).reshape(N, T, self.d_model)

        out = self.out_proj(attn_output)

        return out



@register
class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward, deepnorm_alpha, deepnorm_beta, attn_window=None):
        super().__init__()
        self.kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "deepnorm_alpha": deepnorm_alpha,
            "deepnorm_beta": deepnorm_beta,
            "attn_window": attn_window
        }

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            nhead=nhead,
            qkv_bias=False,
            out_bias=True,
            attn_window=attn_window
        )
        self.ff = GatedMlp(
            d_model,
            hidden_features=dim_feedforward,
            activation=F.silu,
            bias1=False,
            bias2=False,
            multiple_of=1,
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.register_buffer("deepnorm_alpha", torch.tensor(deepnorm_alpha))
        self.reset_parameters()

    def reset_parameters(self):
        db = self.kwargs["deepnorm_beta"]
        d_model = self.kwargs["d_model"]
        torch.nn.init.xavier_normal_(self.ff.fc1.weight, gain=db)
        torch.nn.init.xavier_normal_(self.ff.fc2.weight, gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.out_proj.weight, gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight[2*d_model:], gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight[:2*d_model], gain=1)

    def forward(self, x):
        x = self.norm1(self.self_attn(x), self.deepnorm_alpha*x)
        x = self.norm2(self.ff(x), self.deepnorm_alpha*x)
        return x

    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        return self.kwargs




def use_koi(self, **kwargs):
    # koi needs modified LinearCRFLayer settings
    def _expand_blanks(m):
        if isinstance(m, LinearCRFEncoder):
            m.expand_blanks = False
    self.encoder.apply(_expand_blanks)
    self.encoder = Serial([
        self.encoder,
        Permute([1, 0, 2]),
        MakeContiguous(),
    ])

def Model(config):
    model_config = {k: v for k, v in config["model"].items() if k != "package"}
    model = from_dict(model_config)
    model.config = config
    model.use_koi = types.MethodType(use_koi, model)
    return model

def create_mamba_block(
    d_model,
    d_intermediate,
    d_state,
    d_conv,
    headdim,
    expand,
    chunk_size
    ):
        block = Mamba2(d_model=d_model, d_state=d_state, 
                       headdim=headdim, expand=expand, d_conv=d_conv, chunk_size=chunk_size)
        return block

class Model(Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if 'qscore' not in config:
            self.qkv_bias = 0.0
            self.out_bias = 1.0
        else:
            self.qkv_bias = config['qscore']['bias']
            self.out_bias = config['qscore']['scale']
        self.config = config
        d_model = config['block'][0]['d_model']
        self.d_model = config['block'][0]['d_model']
        self.nhead = config['block'][0]['nhead']
        nhead =self.nhead
        d_intermediate = config['block'][0]['d_intermediate']
        n_layer = config['block'][0]['n_layer']
        d_state = config['block'][0]['d_state']
        d_conv = config['block'][0]['d_conv']
        headdim = config['block'][0]['headdim']
        self.head_dim = headdim
        expand = config['block'][0]['expand']
        vocab_size = config['block'][0]['vocab_size']
        pad_vocab_size = config['block'][0]['pad_vocab_size']
        chunk_size = config['block'][0]['chunk_size']
        dim_feedforward = config['block'][0]['dim_feedforward']
        deepnorm_alpha = config['block'][0]['deepnorm_alpha']
        deepnorm_beta = config['block'][0]['deepnorm_beta']

        assert self.d_model % self.nhead == 0, "d_model must be divisible by nhead"
        self.kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "deepnorm_alpha": deepnorm_alpha,
            "deepnorm_beta": deepnorm_beta,
        }

        self.mamba_layers = ModuleList(
            [
                create_mamba_block(
                    d_model=self.d_model,
                    d_intermediate=d_intermediate,
                    d_state=d_state,
                    d_conv=d_conv,
                    headdim=headdim,
                    expand=expand,
                    chunk_size=chunk_size,
                )
                for i in range(n_layer)
            ]
        )
        self.rotary_dim = self.d_model // self.nhead
        self.rotary_emb = RotaryEmbedding(self.rotary_dim, interleaved=False)
        self.Wqkv = torch.nn.Linear(d_model, 3 * d_model, bias=self.qkv_bias)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=self.out_bias)
        self.ff = GatedMlp(
            d_model,
            hidden_features=dim_feedforward,
            activation=F.silu,
            bias1=False,
            bias2=False,
            multiple_of=1,
        )
        self.norm1 =RMSNorm(d_model)
        self.norm2 =RMSNorm(d_model)

    def forward(self, x):
        N, T, _ = x.shape
        print("check  1 2 3 ",x.shape)
        print("\n \n \n \n")
        qkv = self.Wqkv(x).view(N, T, 3, self.nhead, self.head_dim)

        qkv = self.rotary_emb(qkv)

        mamba_output = self.mamba_layers(qkv).reshape(N, T, self.d_model)

        out = self.out_proj(mamba_output)
        out = self.norm1(x, self.deepnorm_alpha*x)
        out = self.norm2(self.ff(x), self.deepnorm_alpha*x)
        return out
    
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

    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        return self.kwargs
'''
