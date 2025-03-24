from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

@dataclass
class TransformerArgs:
    embedding_dim: int = 2048
    nb_layers: int = 22
    head_dim: int = 64
    exploded_dim: int = 5632
    n_heads: int = 32
    n_kv_heads: int = 4
    eps: float = 1e-6
    vocab_size: int = 32000
    precompute_rotary: int = 256
    dtype: torch.dtype = torch.float16
    norm_dtype: torch.dtype = torch.float32

class RMSNorm(torch.nn.Module):
    def __init__(self, args: TransformerArgs, norm=None):
        super().__init__()
        self.eps = args.eps
        self.dtype = args.norm_dtype
        self.weight = nn.Parameter(
            torch.ones(args.embedding_dim, dtype=args.dtype)
        )
        if norm is not None:
            self.load(norm)

    def forward(self, x):
        output = x.to(self.dtype)
        norm = torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        output = (output*norm).type_as(x)
        return output * self.weight
    
    def load(self, norm):
        self.weight = norm.weight

class Attention(nn.Module):
    def __init__(self, args: TransformerArgs, positional_encoder, attention=None):
        super().__init__()
        self.Hq, self.H = args.n_heads, args.n_kv_heads
        E = args.embedding_dim
        object.__setattr__(self, 'positional_encoder', positional_encoder)
        self.q_proj = nn.Linear(in_features=E, out_features=self.Hq*args.head_dim, bias=False, dtype=args.dtype)
        self.k_proj = nn.Linear(in_features=E, out_features=self.H*args.head_dim, bias=False, dtype=args.dtype)
        self.v_proj = nn.Linear(in_features=E, out_features=self.H*args.head_dim, bias=False, dtype=args.dtype)
        self.o_proj = nn.Linear(in_features=E, out_features=E, bias=False, dtype=args.dtype)
        if attention is not None:
            self.load(attention)

    def forward(self, x):
        N, S, E = x.shape
        xq = self.q_proj(x).view(N, S, self.Hq, -1)
        xk = self.k_proj(x).view(N, S, self.H, -1)
        xv = self.v_proj(x).view(N, S, self.H, -1)
        xq = self.positional_encoder(xq)
        xk = self.positional_encoder(xk)
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
    def __init__(self, args: TransformerArgs, mlp=None):
        super().__init__()
        self.gate_proj = nn.Linear(in_features=args.embedding_dim, out_features=args.exploded_dim, bias=False, dtype=args.dtype)
        self.up_proj = nn.Linear(in_features=args.embedding_dim, out_features=args.exploded_dim, bias=False, dtype=args.dtype)
        self.down_proj = nn.Linear(in_features=args.exploded_dim, out_features=args.embedding_dim, bias=False, dtype=args.dtype)
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
    def __init__(self, args: TransformerArgs, positional_encoder, block=None):
        super().__init__()
        attn, ffn, n1, n2 = [None]*4 if block is None else (block.attention, block.feed_forward, block.attention_norm, block.ffn_norm)
        self.attention = Attention(args, positional_encoder, attn)
        self.feed_forward = FeedForward(args, ffn)
        self.attention_norm = RMSNorm(args, n1)
        self.ffn_norm = RMSNorm(args, n2)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class Transformers(nn.Module):
    def __init__(self, args: TransformerArgs, model=None):
        super().__init__()
        self.positional_encoder = RotaryPositionalEmbeddings(args.head_dim, args.precompute_rotary)
        if model is not None:
            self.load_from_mistral(model, args)
        else:
            self.embeddings = nn.Embedding(args.vocab_size, args.embedding_dim, dtype=args.dtype)
            self.layers = nn.ModuleList(
                [Block(args, self.positional_encoder) for _ in range(args.nb_layers)]
            )
            self.norm = RMSNorm(args)
            self.output = nn.Linear(in_features=args.embedding_dim, out_features=args.vocab_size, bias=False, dtype=args.dtype)

    def load_from_mistral(self, model, args):
        self.embeddings = model.tok_embeddings
        self.layers = nn.ModuleList(
            [Block(args, self.positional_encoder, block) for block in model.layers]
        )
        self.norm = RMSNorm(args, norm=model.norm)
        self.output = model.output

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))