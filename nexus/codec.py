"""
nexus/codec.py
NibbleNet: the Nexus byte-position codec.

Implementation of the byte-level structured embedding proposed as
"Kronecker Embeddings" by Rohan Shravan (2026); renamed NibbleNet in this
codebase. Cite the original paper for the method.

Implements exactly the encoding from the proposal:
    kappa(b) = (1/sqrt(L)) * sum_p  c_{b_p} (x) p_p
with dc = 256 byte values, dp = max byte positions, D = 256 * dp.
Linearized one-hot index for (byte v at position p) is  v * dp + p.

Everything here is framework-minimal on purpose so the same file runs
unchanged on Colab T4 and on EC2.
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Byte-level BPE unicode mapping (GPT-2 standard). GPT-2, SmolLM2, Pythia and
# Qwen tokenizers all store token strings through this same mapping, so
# inverting it recovers the token's true raw bytes.
# ---------------------------------------------------------------------------
def _bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_UNI2BYTE = {v: k for k, v in _bytes_to_unicode().items()}


def token_to_bytes(token: str) -> bytes:
    """Recover the raw UTF-8 bytes of a tokenizer piece.

    Handles: SentencePiece byte-fallback (<0xNN>), SentencePiece space marker
    (\\u2581), byte-level-BPE unicode remapping (GPT-2 family), and plain
    strings / special tokens (literal surface bytes, as in the paper).
    """
    # byte-fallback token like <0xC3>
    if len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
        try:
            return bytes([int(token[3:5], 16)])
        except ValueError:
            pass
    # byte-level BPE: every char maps back through the GPT-2 unicode table
    if token and all(c in _UNI2BYTE for c in token):
        return bytes(_UNI2BYTE[c] for c in token)
    # SentencePiece style or special tokens: literal surface form
    return token.replace("\u2581", " ").encode("utf-8", errors="replace")


def utf8_safe_truncate(b: bytes, dp: int) -> bytes:
    """Truncate to dp bytes, backing off if the cut lands mid-codepoint."""
    if len(b) <= dp:
        return b
    cut = dp
    while cut > 0 and (b[cut] & 0xC0) == 0x80:
        cut -= 1
    return b[:cut] if cut > 0 else b[:dp]


def build_byte_buffer(tokenizer, dp: int):
    """Return (tok_bytes uint8 [V, dp], tok_len int16 [V]) for a HF tokenizer."""
    vocab = tokenizer.get_vocab()
    V = max(vocab.values()) + 1
    tok_bytes = torch.zeros(V, dp, dtype=torch.uint8)
    tok_len = torch.zeros(V, dtype=torch.int16)
    for tok, idx in vocab.items():
        b = utf8_safe_truncate(token_to_bytes(tok), dp)
        L = max(len(b), 1)  # never allow a zero-length token
        if len(b) == 0:
            b = b"\x00"
        tok_bytes[idx, : len(b)] = torch.tensor(list(b), dtype=torch.uint8)
        tok_len[idx] = L
    return tok_bytes, tok_len


def codec_from_bytes(byte_seqs, dp: int, znorm: bool = False) -> torch.Tensor:
    """Compute kappa(b) for a list of python `bytes` objects. Returns [N, D].

    Used by the probe scripts (claim 2) where we want raw codec vectors for
    arbitrary strings, including OOV strings never seen by any tokenizer.
    """
    D = 256 * dp
    out = torch.zeros(len(byte_seqs), D)
    for i, b in enumerate(byte_seqs):
        b = utf8_safe_truncate(b, dp)
        L = max(len(b), 1)
        for p, v in enumerate(b):
            out[i, v * dp + p] += 1.0 / math.sqrt(L)
    if znorm:
        out = (out - out.mean(-1, keepdim=True)) / (out.std(-1, keepdim=True) + 1e-6)
    return out


class NibbleNetEmbedding(nn.Module):
    """Drop-in replacement for nn.Embedding: fixed byte-position codec
    (gpu_dynamic variant, recomputed each forward) + one learned projection.

    Only trainable parameter: proj (D -> d_model), init N(0, 1/sqrt(D)).
    """

    def __init__(self, tok_bytes: torch.Tensor, tok_len: torch.Tensor,
                 d_model: int, dp: int):
        super().__init__()
        self.dp = dp
        self.D = 256 * dp
        self.register_buffer("tok_bytes", tok_bytes, persistent=True)
        self.register_buffer("tok_len", tok_len, persistent=True)
        self.proj = nn.Linear(self.D, d_model, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1.0 / math.sqrt(self.D))

    def codec(self, idx: torch.Tensor) -> torch.Tensor:
        """idx [..] int64 -> z-normalized codec vectors [.., D] (float32)."""
        b = self.tok_bytes[idx].long()                      # [.., dp]
        L = self.tok_len[idx].long()                        # [..]
        pos = torch.arange(self.dp, device=idx.device)      # [dp]
        mask = pos < L.unsqueeze(-1)                        # [.., dp]
        lin = (b * self.dp + pos) * mask                    # masked -> slot 0
        vals = mask.float() / torch.sqrt(L.float()).unsqueeze(-1)
        k = torch.zeros(*idx.shape, self.D, device=idx.device)
        k.scatter_add_(-1, lin, vals)
        # per-token z-normalization (proposal: zero mean, unit variance)
        k = (k - k.mean(-1, keepdim=True)) / (k.std(-1, keepdim=True) + 1e-6)
        return k

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.proj(self.codec(idx))
