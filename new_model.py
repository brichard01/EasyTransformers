import torch
from torch import nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

positional_encoder = RotaryPositionalEmbeddings(64, 20)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, norm=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        if norm is not None:
            self.load(norm)

    def forward(self, x):
        output = x.float()
        norm = torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        output = (output*norm).type_as(x)
        return output * self.weight
    
    def load(self, norm):
        self.weight = norm.weight

class Attention(nn.Module):
    def __init__(self, attention=None):
        super().__init__()
        self.q_proj = nn.Linear(in_features=2048, out_features=2048, bias=False)
        self.k_proj = nn.Linear(in_features=2048, out_features=256, bias=False)
        self.v_proj = nn.Linear(in_features=2048, out_features=256, bias=False)
        self.o_proj = nn.Linear(in_features=2048, out_features=2048, bias=False)
        self.Hq, self.H = 32, 4
        if attention is not None:
            self.load(attention)

    def forward(self, x):
        N, S, E = x.shape
        xq = self.q_proj(x).view(N, S, self.Hq, -1)
        xk = self.k_proj(x).view(N, S, self.H, -1)
        xv = self.v_proj(x).view(N, S, self.H, -1)
        xq = positional_encoder(xq)
        xk = positional_encoder(xk)
        attention = F.scaled_dot_product_attention(
            xq.transpose(1, 2),
            xk.transpose(1, 2),
            xv.transpose(1, 2),
            enable_gqa=True,
            is_causal=True,
        )
        return self.o_proj(attention.transpose(1, 2).contiguous().view(N, S, E))
    
    def load(self, attention):
        self.q_proj.weight = attention.wq.weight
        self.k_proj.weight = attention.wk.weight
        self.v_proj.weight = attention.wv.weight
        self.o_proj.weight = attention.wo.weight

class FeedForward(nn.Module):
    def __init__(self, mlp=None):
        super().__init__()
        self.gate_proj = nn.Linear(in_features=2048, out_features=5632, bias=False)
        self.up_proj = nn.Linear(in_features=2048, out_features=5632, bias=False)
        self.down_proj = nn.Linear(in_features=5632, out_features=2048, bias=False)
        self.act_fn = nn.SiLU()
        if mlp is not None:
            self.load(mlp)

    def forward(self, x):
        x = self.up_proj(x)*self.act_fn(self.gate_proj(x))
        return self.down_proj(x)
    
    def load(self, mlp):
        self.gate_proj.weight = mlp.w1.weight
        self.up_proj.weight = mlp.w3.weight
        self.down_proj.weight = mlp.w2.weight

class Block(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.attention = Attention(block.attention)
        self.feed_forward = FeedForward(block.feed_forward)
        self.attention_norm = RMSNorm(2048, norm=block.attention_norm)
        self.ffn_norm = RMSNorm(2048, norm=block.ffn_norm)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class Transformers(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embeddings = model.tok_embeddings
        self.layers = nn.ModuleList(
            [Block(block) for block in model.layers]
        )
        self.norm = RMSNorm(2048, norm=model.norm)
        self.output = model.output

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))