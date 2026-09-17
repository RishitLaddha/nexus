"""
nexus/model.py
Minimal nanoGPT-style GPT. The ONLY thing that differs between the two arms
is the input embedding pathway, exactly as in the paper's controlled setup:

  arm = "bpe"  : learned nn.Embedding, weight-tied to lm_head (BPE-tied)
  arm = "nibble" : fixed byte-position codec + learned D->d projection,
                 lm_head untied (tying is architecturally inapplicable)

Everything else (attention, MLP, LN, learned positions, loss) is identical.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import NibbleNetEmbedding


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    embedding: str = "bpe"   # "bpe" (tied) or "nibble"
    dp: int = 16             # codec max byte positions (D = 256*dp = 4096)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig, tok_bytes=None, tok_len=None):
        super().__init__()
        self.cfg = cfg
        if cfg.embedding == "nibble":
            assert tok_bytes is not None and tok_len is not None
            self.wte = NibbleNetEmbedding(tok_bytes, tok_len, cfg.n_embd, cfg.dp)
        else:
            self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
            nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        nn.init.normal_(self.wpe.weight, mean=0.0, std=0.02)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.embedding == "bpe":
            self.lm_head.weight = self.wte.weight   # weight tying (BPE-tied arm)
        else:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        # scaled init on residual projections (nanoGPT convention)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * cfg.n_layer))

    def embed(self, idx):
        """Input-pathway output only (Eraw for the layered probe)."""
        return self.wte(idx)

    def forward(self, idx, targets=None, return_hidden=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        hiddens = []
        for block in self.h:
            x = block(x)
            if return_hidden:
                hiddens.append(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        if return_hidden:
            return logits, loss, hiddens
        return logits, loss

    # ----- accounting helpers (claim 3 parameter table) -----
    def param_report(self):
        cfg = self.cfg
        total = sum(p.numel() for p in self.parameters())
        if cfg.embedding == "bpe":
            emb = self.wte.weight.numel()
            rep = {"arm": "bpe-tied", "input_side_trainable": emb,
                   "lm_head": "tied (same block)",
                   "total_trainable": total}
        else:
            proj = self.wte.proj.weight.numel()
            buf = self.wte.tok_bytes.numel() + 2 * self.wte.tok_len.numel()
            rep = {"arm": "nibble", "input_side_trainable": proj,
                   "codec_buffer_bytes": buf,
                   "lm_head": self.lm_head.weight.numel(),
                   "total_trainable": total}
        return rep
