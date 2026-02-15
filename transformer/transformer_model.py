from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from typing import Optional

mistral_mapping = {
    'q_proj': 'wq',
    'k_proj': 'wk',
    'v_proj': 'wv',
    'o_proj': 'wo',
    'gate_proj': 'w1',
    'down_proj': 'w2',
    'up_proj': 'w3',
    'attention': 'attention',
    'feed_forward': 'feed_forward',
    'attention_norm': 'attention_norm',
    'ffn_norm': 'ffn_norm',
}

transformers_mapping = {
    'q_proj': 'q_proj',
    'k_proj': 'k_proj',
    'v_proj': 'v_proj',
    'o_proj': 'o_proj',
    'gate_proj': 'gate_proj',
    'down_proj': 'down_proj',
    'up_proj': 'up_proj',
    'attention': 'self_attn',
    'feed_forward': 'mlp',
    'attention_norm': 'input_layernorm',
    'ffn_norm': 'post_attention_layernorm',
}

@dataclass
class TransformerArgs:
    embedding_dim: int = 2048
    nb_layers: int = 22
    head_dim: int = 64
    exploded_dim: int = 5632
    n_heads: int = 32
    n_kv_heads: int = 4
    vocab_size: int = 32000
    max_seq_len: int = 256
    rope_base: int = 10000
    pairing_style: Optional[str] = None
    dtype: torch.dtype = torch.float16
    norm_dtype: torch.dtype = torch.float32
    norm_eps: float = 1e-6

class RMSNorm(torch.nn.Module):
    def __init__(self, args: TransformerArgs, norm=None):
        super().__init__()
        self.eps = args.norm_eps
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

class RoPE(RotaryPositionalEmbeddings):
    def __init__(self, args, pairing_style='mistral_inf'):
        super().__init__(args.head_dim, args.max_seq_len, args.rope_base)
        self.pairing_style = args.pairing_style
        if self.pairing_style is None:
            self.pairing_style = pairing_style

    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None):
        if self.pairing_style == 'mistral_inf':
            x = super().forward(x, input_pos=input_pos)
        else:
            shape = x.shape
            x = torch.stack(torch.split(x, shape[-1]//2, dim=-1), dim=-1).view(*shape)
            x = super().forward(x, input_pos=input_pos)
            x = x.view(*shape[:-1], -1, 2).transpose(3, 4).reshape(shape)
        return x

class Attention(nn.Module):
    def __init__(self, args: TransformerArgs, positional_encoder, attention=None, mapping=None):
        super().__init__()
        self.Hq, self.H = args.n_heads, args.n_kv_heads
        E = args.embedding_dim
        object.__setattr__(self, 'positional_encoder', positional_encoder)
        self.q_proj = nn.Linear(in_features=E, out_features=self.Hq*args.head_dim, bias=False, dtype=args.dtype)
        self.k_proj = nn.Linear(in_features=E, out_features=self.H*args.head_dim, bias=False, dtype=args.dtype)
        self.v_proj = nn.Linear(in_features=E, out_features=self.H*args.head_dim, bias=False, dtype=args.dtype)
        self.o_proj = nn.Linear(in_features=E, out_features=E, bias=False, dtype=args.dtype)
        if attention is not None:
            self.load(attention, mapping)
        self.cache = None

    def forward(self, x):
        N, S, E = x.shape
        xq = self.q_proj(x).view(N, S, self.Hq, -1)
        xk = self.k_proj(x).view(N, S, self.H, -1)
        xv = self.v_proj(x).view(N, S, self.H, -1)
        xq = self.positional_encoder(xq)
        xk = self.positional_encoder(xk)
        if self.cache:
            self.cache(xk, xv)
        attention = F.scaled_dot_product_attention(
            xq.transpose(1, 2),
            xk.transpose(1, 2),
            xv.transpose(1, 2),
            enable_gqa=True,
            is_causal=True,
        )
        return self.o_proj(attention.transpose(1, 2).contiguous().view(N, S, E))
    
    def forward_cache(self, x):
        N, S, E = x.shape
        xq = self.q_proj(x).view(N, S, self.Hq, -1)
        xk = self.k_proj(x).view(N, S, self.H, -1)
        xv = self.v_proj(x).view(N, S, self.H, -1)
        position = self.cache.v.shape[1]
        position = torch.tensor([position], dtype=torch.int64)
        xq = self.positional_encoder(xq, position)
        xk = self.positional_encoder(xk, position)
        xk, xv = self.cache(xk, xv)
        attention = F.scaled_dot_product_attention(
            xq.transpose(1, 2),
            xk.transpose(1, 2),
            xv.transpose(1, 2),
            enable_gqa=True,
            is_causal=False,
        )
        return self.o_proj(attention.transpose(1, 2).contiguous().view(N, S, E))
    
    def load(self, attention, mapping):
        self.q_proj.weight = getattr(attention, mapping['q_proj']).weight
        self.k_proj.weight = getattr(attention, mapping['k_proj']).weight
        self.v_proj.weight = getattr(attention, mapping['v_proj']).weight
        self.o_proj.weight = getattr(attention, mapping['o_proj']).weight

class FeedForward(nn.Module):
    def __init__(self, args: TransformerArgs, mlp=None, mapping=None):
        super().__init__()
        self.gate_proj = nn.Linear(in_features=args.embedding_dim, out_features=args.exploded_dim, bias=False, dtype=args.dtype)
        self.up_proj = nn.Linear(in_features=args.embedding_dim, out_features=args.exploded_dim, bias=False, dtype=args.dtype)
        self.down_proj = nn.Linear(in_features=args.exploded_dim, out_features=args.embedding_dim, bias=False, dtype=args.dtype)
        self.act_fn = nn.SiLU()
        if mlp is not None:
            self.load(mlp, mapping)

    def forward(self, x):
        x = self.up_proj(x)*self.act_fn(self.gate_proj(x))
        return self.down_proj(x)
    
    def load(self, mlp, mapping):
        self.gate_proj.weight = getattr(mlp, mapping['gate_proj']).weight
        self.up_proj.weight = getattr(mlp, mapping['up_proj']).weight
        self.down_proj.weight = getattr(mlp, mapping['down_proj']).weight

class Block(nn.Module):
    def __init__(self, args: TransformerArgs, positional_encoder, block=None, mapping=None):
        super().__init__()
        attn, ffn, n1, n2 = [None]*4
        if block is not None:
            attn, ffn, n1, n2 = (
                getattr(block, mapping['attention']),
                getattr(block, mapping['feed_forward']),
                getattr(block, mapping['attention_norm']),
                getattr(block, mapping['ffn_norm']),
            )
        self.attention = Attention(args, positional_encoder, attn, mapping=mapping)
        self.feed_forward = FeedForward(args, ffn, mapping=mapping)
        self.attention_norm = RMSNorm(args, n1)
        self.ffn_norm = RMSNorm(args, n2)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x
    
    def forward_cache(self, x):
        x = x + self.attention.forward_cache(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x
    
class kv_cache:
    def __init__(self, heads, head_dim, dtype=torch.float16):
        self.heads = heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.resest()   

    def resest(self):
        self.k = torch.zeros(1, 0, self.heads, self.head_dim, dtype=self.dtype)
        self.v = torch.zeros(1, 0, self.heads, self.head_dim, dtype=self.dtype)

    def __call__(self, xk, xv):
        self.k = torch.cat([self.k, xk], 1)
        self.v = torch.cat([self.v, xv], 1)
        return self.k, self.v

class Transformers(nn.Module):
    def __init__(self, args: TransformerArgs, model=None):
        super().__init__()
        self.args = args
        self.positional_encoder = RoPE(args)
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
            [Block(args, self.positional_encoder, block, mapping=mistral_mapping) for block in model.layers]
        )
        self.norm = RMSNorm(args, norm=model.norm)
        self.output = model.output

    def load_from_transformers(self, llm, args):
        self.positional_encoder.pairing_style = 'transformers'
        model = llm.model
        self.embeddings = model.embed_tokens
        self.layers = nn.ModuleList(
            [Block(args, self.positional_encoder, block, mapping=transformers_mapping) for block in model.layers]
        )
        self.norm = RMSNorm(args, norm=model.norm)
        self.output = llm.lm_head

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))
    
    def init_kv_cache(self):
        for layer in self.layers:
            layer.attention.cache = kv_cache(
                self.args.n_kv_heads,
                self.args.head_dim,
                self.args.dtype,
            )

    def step_cache(self, input_ids):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer.forward_cache(x)
        x = x[:, -1, :]
        logits = self.output(self.norm(x))
        return logits
    
    def generate_step(self, input_ids, temperature=1, top_k=10):
        N, _ = input_ids.shape
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = x[:, -1, :]
        logits = self.output(self.norm(x))
        probs = F.softmax(logits/temperature, dim=-1)
        probs, ids = torch.topk(probs, k=top_k)
        probs = (probs / probs.sum(dim=-1).view(N, 1)).cumsum(dim=-1)
        p = torch.rand(N, device=input_ids.device).view(N, 1)
        index = top_k-(p<probs).sum(-1)
        new_ids = torch.Tensor([ids[i, id] for i, id in enumerate(index)]).view(N, 1)
        return torch.cat((input_ids, new_ids.to(input_ids.device, dtype=input_ids.dtype)), dim=-1)


