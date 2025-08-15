import logging
import types
from functools import lru_cache, partial

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import LayerNorm
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

from bonito.crf.model import SeqdistModel
from bonito.nn import from_dict, register, LinearCRFEncoder, MakeContiguous, Module, Permute, Serial

import mamba_ssm
from mamba_ssm import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.block import Block

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

@register
class MultiHeadAttention(Module):
    def __init__(self, d_model, nhead, rotary_dim=None):
        #factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_dim = self.head_dim if rotary_dim is None else rotary_dim
        #self.Wqkv = torch.nn.Linear(d_model, 3 * d_model) #
        mixer_cls = partial(MHA, num_heads=nhead, head_dim=self.head_dim)
        mlp_cls = nn.Identity
        norm_cls = partial(RMSNorm)
        self.block = Block(
            d_model,
            mixer_cls,
            mlp_cls,
            norm_cls,
        )
        self.norm_f = RMSNorm(
            d_model, eps= 1e-5
        )

    def forward(self, x):
        #N, T, _ = x.shape
        #print("\n \n \n \n", x.shape)
        hidden_states, residual = self.block(x)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        return hidden_states


#removed qkv_bias=False, , out_bias=True, rotary_dim=None and wkqv ouptut layer and emds
@register
class MambaBlock(Module):
    def __init__(self, d_model, nhead, d_state, headdim, d_conv, chunk_size):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        #self.rotary_dim = self.head_dim if rotary_dim is None else rotary_dim

        #self.Wqkv = torch.nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        #self.out_proj = torch.nn.Linear(d_model, d_model, bias=out_bias)

        #self.rotary_emb = RotaryEmbedding(self.rotary_dim, interleaved=False)
        #self.attn_window = (-1, -1) if attn_window is None else tuple(attn_window)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            d_conv=d_conv,
            chunk_size=chunk_size,
            )

    def forward(self, x):
            #N, T, _ = x.shape
            #print("\n \n \n \n", x.shape)
            #qkv = self.Wqkv(x).view(N, T, 3, self.nhead, self.head_dim)
            #print("\n \n \n \n", qkv.shape)
            #qkv = self.rotary_emb(x.view(N, T, 3, self.nhead, self.head_dim))
            if isinstance(x, tuple):
                x = torch.cat(x, dim=-1)
            out = self.mamba(x)

            #out = self.out_proj(mamba_output)

            return out
    
# qkv_bias=False,
@register
class MambaLayer(Module):
    def __init__(self, d_model, nhead, nlayer, dim_feedforward, deepnorm_alpha, deepnorm_beta,chunk_size, d_state=128, headdim=64, d_conv=4):
        super().__init__()
        self.kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "nlayer": nlayer,
            "dim_feedforward": dim_feedforward,
            "deepnorm_alpha": deepnorm_alpha,
            "deepnorm_beta": deepnorm_beta,
            "d_state": d_state,
            "headdim": headdim,
            "d_conv": d_conv,
        }
        self.self_attn = MultiHeadAttention(d_model=d_model, nhead=nhead)
        # maybe switch to nn.embedding?
        # added an MHA

        self.mamba = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    nhead=nhead,
                    d_state=d_state,
                    headdim=headdim,
                    d_conv=d_conv,
                    chunk_size=chunk_size,
                    #out_bias=True,            
                )
            for i in range(nlayer)
            ]
        )
        self.ff = GatedMlp(
            d_model,
            hidden_features=dim_feedforward,
            activation=F.silu,
            bias1=False,
            bias2=False,
            multiple_of=1,
        )
        #self.norm1 = RMSNorm(d_model)
        #self.norm2 = RMSNorm(d_model)
        self.norm = RMSNorm(d_model)

        self.register_buffer("deepnorm_alpha", torch.tensor(deepnorm_alpha))
        self.reset_parameters()

    def reset_parameters(self):
        db = self.kwargs["deepnorm_beta"]
        d_model = self.kwargs["d_model"]
        torch.nn.init.xavier_normal_(self.ff.fc1.weight, gain=db)
        torch.nn.init.xavier_normal_(self.ff.fc2.weight, gain=db)
        #torch.nn.init.xavier_normal_(self.mamba.out_proj.weight, gain=db)
        #torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight, gain=db)
        #torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight[:2*d_model], gain=1)

    def forward(self, x):
        #print("\n \n \n \n", x.shape)
        x = self.self_attn(x)
        #print("\n \n \n \n", x.shape)
        #x = self.norm1(self.mamba(x), self.deepnorm_alpha*x)
        for layer in self.mamba:
            x = layer(x)

        x = self.norm(self.ff(x), self.deepnorm_alpha*x)
        return x

    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        return self.kwargs

#added embeddings
@register
class Mamba2Block(Module):
    def __init__(self, d_model, nhead, d_state, headdim, d_conv, chunk_size, deepnorm_alpha, rotary_dim=None):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.deepnorm_alpha = deepnorm_alpha
        self.rotary_dim = self.head_dim if rotary_dim is None else rotary_dim
        self.input_proj = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.rotary_emb = RotaryEmbedding(self.rotary_dim, interleaved=False)
        mamba_cls = partial(Mamba2,
            d_state=d_state,
            headdim=headdim,
            d_conv=d_conv,
            chunk_size=chunk_size,
            )
        mlp_cls = partial(nn.Identity)
        norm_cls = partial(RMSNorm, eps=1e-5)
        self.mamba = Block(
            d_model,
            mamba_cls,
            mlp_cls,
            norm_cls,
        )


    def forward(self, x):
        """
        if isinstance(x, tuple):
            x = torch.cat(x, dim=-1)
        """
        """
        hidden_states, residual = self.mamba(x)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        return residual
        """
        N, T, _ = x.shape
        qkv = self.input_proj(x).view(N, T, 3, self.nhead, self.head_dim)
        qkv_rotary = self.rotary_emb(qkv)
        
        # Extract Q component for Mamba input
        q_rotary = qkv_rotary[:, :, 0, :, :].reshape(N, T, self.d_model)
        
        hidden_states, residual = self.mamba(q_rotary)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        return residual
    
    def reset_parameters(self, gain=1.0):
        # Recursively initialize parameters
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, gain=gain)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                torch.nn.init.ones_(module.weight)

#removed embeddings
@register
class Mamba2Blockv2(Module):
    def __init__(self, d_model, nhead, d_state, headdim, d_conv, chunk_size, deepnorm_alpha, rotary_dim=None):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.deepnorm_alpha = deepnorm_alpha
        mamba_cls = partial(Mamba2,
            d_state=d_state,
            headdim=headdim,
            d_conv=d_conv,
            chunk_size=chunk_size,
            )
        mlp_cls = partial(nn.Identity)
        norm_cls = partial(RMSNorm, eps=1e-5)
        self.mamba = Block(
            d_model,
            mamba_cls,
            mlp_cls,
            norm_cls,
        )


    def forward(self, x):
        """
        if isinstance(x, tuple):
            x = torch.cat(x, dim=-1)
        """
        
        hidden_states, residual = self.mamba(x)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        return residual
        
        
    
    def reset_parameters(self, gain=1.0):
        # Recursively initialize parameters
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, gain=gain)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                torch.nn.init.ones_(module.weight)
# rotary embedding version
@register
class MambaLayer2(Module):
    def __init__(self, d_model, nhead, nlayer, dim_feedforward, deepnorm_alpha, deepnorm_beta,chunk_size, d_state=128, headdim=64, d_conv=4):
        super().__init__()
        self.kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "nlayer": nlayer,
            "dim_feedforward": dim_feedforward,
            "deepnorm_alpha": deepnorm_alpha,
            "deepnorm_beta": deepnorm_beta,
            "d_state": d_state,
            "headdim": headdim,
            "d_conv": d_conv,
        }
        # self.self_attn = MultiHeadAttention(d_model=d_model, nhead=nhead)
        # maybe switch to nn.embedding?
        # added an MHA
        
        self.mamba = nn.ModuleList(
            [
                Mamba2Block(
                    d_model=d_model,
                    nhead=nhead,
                    d_state=d_state,
                    headdim=headdim,
                    d_conv=d_conv,
                    chunk_size=chunk_size,
                    deepnorm_alpha=deepnorm_alpha,
                    #out_bias=True,            
                )
            for i in range(nlayer)
            ]
        )
        self.ff = GatedMlp(
            d_model,
            hidden_features=dim_feedforward,
            activation=F.silu,
            bias1=False,
            bias2=False,
            multiple_of=1,
        )
        
        self.norm = RMSNorm(d_model)

        self.register_buffer("deepnorm_alpha", torch.tensor(deepnorm_alpha))
        self.reset_parameters()

    def reset_parameters(self):
        db = self.kwargs["deepnorm_beta"]
        d_model = self.kwargs["d_model"]
        torch.nn.init.xavier_normal_(self.ff.fc1.weight, gain=db)
        torch.nn.init.xavier_normal_(self.ff.fc2.weight, gain=db)
        for layer in self.mamba:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters(gain=db)


    def forward(self, x):
        #print("\n \n \n \n", x.shape)
        # x = self.self_attn(x)
        #print("\n \n \n \n", x.shape)
        #x = self.norm1(self.mamba(x), self.deepnorm_alpha*x)
        for layer in self.mamba:
            x = layer(x)

        x = self.norm(self.ff(x))
        return x

    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        return self.kwargs



# no embedding version
@register
class Mamba2Layer(Module):
    def __init__(self, d_model, nhead, nlayer, dim_feedforward, deepnorm_alpha, deepnorm_beta,chunk_size, d_state=128, headdim=64, d_conv=4):
        super().__init__()
        self.kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "nlayer": nlayer,
            "dim_feedforward": dim_feedforward,
            "deepnorm_alpha": deepnorm_alpha,
            "deepnorm_beta": deepnorm_beta,
            "d_state": d_state,
            "headdim": headdim,
            "d_conv": d_conv,
        }
        # self.self_attn = MultiHeadAttention(d_model=d_model, nhead=nhead)
        # maybe switch to nn.embedding?
        # added an MHA
        
        self.mamba = nn.ModuleList(
            [
                Mamba2Blockv2(
                    d_model=d_model,
                    nhead=nhead,
                    d_state=d_state,
                    headdim=headdim,
                    d_conv=d_conv,
                    chunk_size=chunk_size,
                    deepnorm_alpha=deepnorm_alpha,
                    #out_bias=True,            
                )
            for i in range(nlayer)
            ]
        )
        self.ff = GatedMlp(
            d_model,
            hidden_features=dim_feedforward,
            activation=F.silu,
            bias1=False,
            bias2=False,
            multiple_of=1,
        )
        
        self.norm = RMSNorm(d_model)

        self.register_buffer("deepnorm_alpha", torch.tensor(deepnorm_alpha))
        self.reset_parameters()

    def reset_parameters(self):
        db = self.kwargs["deepnorm_beta"]
        d_model = self.kwargs["d_model"]
        torch.nn.init.xavier_normal_(self.ff.fc1.weight, gain=db)
        torch.nn.init.xavier_normal_(self.ff.fc2.weight, gain=db)
        for layer in self.mamba:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters(gain=db)


    def forward(self, x):
        #print("\n \n \n \n", x.shape)
        # x = self.self_attn(x)
        #print("\n \n \n \n", x.shape)
        #x = self.norm1(self.mamba(x), self.deepnorm_alpha*x)
        for layer in self.mamba:
            x = layer(x)

        x = self.norm(self.ff(x))
        return x

    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        return self.kwargs
    
#added embeddings and GatedMlp
@register
class Mamba2Blockv3(Module):
    def __init__(self, d_model, nhead, d_state, headdim, d_conv, chunk_size, deepnorm_alpha, dim_feedforward, rotary_dim=None):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.deepnorm_alpha = deepnorm_alpha
        self.rotary_dim = self.head_dim if rotary_dim is None else rotary_dim
        self.input_proj = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.rotary_emb = RotaryEmbedding(self.rotary_dim, interleaved=False)
        mamba_cls = partial(Mamba2,
            d_state=d_state,
            headdim=headdim,
            d_conv=d_conv,
            chunk_size=chunk_size,
            )
        mlp_cls = partial(nn.Identity)
        norm_cls = partial(RMSNorm, eps=1e-5)
        self.mamba = Block(
            d_model,
            mamba_cls,
            mlp_cls,
            norm_cls,
        )
        self.ff = GatedMlp(
            d_model,
            hidden_features=dim_feedforward,
            activation=F.silu,
            bias1=False,
            bias2=False,
            multiple_of=1,
        )
        self.norm = RMSNorm(d_model)


    def forward(self, x):
        """
        if isinstance(x, tuple):
            x = torch.cat(x, dim=-1)
        """
        """
        hidden_states, residual = self.mamba(x)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        return residual
        """
        N, T, _ = x.shape
        qkv = self.input_proj(x).view(N, T, 3, self.nhead, self.head_dim)
        qkv_rotary = self.rotary_emb(qkv)
        
        # Extract Q component for Mamba input
        q_rotary = qkv_rotary[:, :, 0, :, :].reshape(N, T, self.d_model)
        
        hidden_states, residual = self.mamba(q_rotary)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        residual = self.norm(self.ff(residual))
        return residual
    
    def reset_parameters(self, gain=1.0):
        # Recursively initialize parameters
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, gain=gain)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                torch.nn.init.ones_(module.weight)
        torch.nn.init.xavier_normal_(self.ff.fc1.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.ff.fc2.weight, gain=gain)

# rotary embedding version
@register
class MambaLayer2v3(Module):
    def __init__(self, d_model, nhead, nlayer, dim_feedforward, deepnorm_alpha, deepnorm_beta,chunk_size, d_state=128, headdim=64, d_conv=4):
        super().__init__()
        self.kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "nlayer": nlayer,
            "dim_feedforward": dim_feedforward,
            "deepnorm_alpha": deepnorm_alpha,
            "deepnorm_beta": deepnorm_beta,
            "d_state": d_state,
            "headdim": headdim,
            "d_conv": d_conv,
        }
        # self.self_attn = MultiHeadAttention(d_model=d_model, nhead=nhead)
        # maybe switch to nn.embedding?
        # added an MHA
        
        self.mamba = nn.ModuleList(
            [
                Mamba2Blockv3(
                    d_model=d_model,
                    nhead=nhead,
                    d_state=d_state,
                    headdim=headdim,
                    d_conv=d_conv,
                    chunk_size=chunk_size,
                    dim_feedforward=dim_feedforward,
                    deepnorm_alpha=deepnorm_alpha,
                    #out_bias=True,            
                )
            for i in range(nlayer)
            ]
        )
        
        self.register_buffer("deepnorm_alpha", torch.tensor(deepnorm_alpha))
        self.reset_parameters()

    def reset_parameters(self):
        db = self.kwargs["deepnorm_beta"]
        d_model = self.kwargs["d_model"]
        
        for layer in self.mamba:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters(gain=db)


    def forward(self, x):
        #print("\n \n \n \n", x.shape)
        # x = self.self_attn(x)
        #print("\n \n \n \n", x.shape)
        #x = self.norm1(self.mamba(x), self.deepnorm_alpha*x)
        for layer in self.mamba:
            x = layer(x)
        
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