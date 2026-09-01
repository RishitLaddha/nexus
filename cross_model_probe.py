"""
Cross-model embedding-geometry probe (paper Sections 6.1–6.4).

For each Hugging Face language model in ``--models``, this probe answers
two questions about the model's input embedding table E (or its
nexus-codec equivalent):

  1. How well does E reflect orthographic structure?
     - For a set of (anchor, neighbor) string pairs that differ by
       1-2 character edits, compute cosine(E[anchor], E[neighbor]).
     - Tokenizers that segment characters into separate tokens (BPE) lose
       this structure; byte-level codecs preserve it.

  2. How well does E reflect semantic / categorical structure?
     - For a set of category clusters (e.g. {dog, cat, horse, sheep}),
       compute mean within-cluster cosine and mean between-cluster cosine.
     - Quotient = within / between, higher = more separation.

The probe is general-purpose: you pass it a list of HF model ids
(``meta-llama/Llama-3.2-1B``, ``Qwen/Qwen3-8B``, etc.) and it loops over
each. Each model's tokenizer is used for both metrics — for multi-token
strings the probe pools the constituent tokens' embeddings (mean).

Usage::

    python probes/cross_model_probe.py \\
        --models meta-llama/Llama-3.2-1B Qwen/Qwen3-8B google/gemma-3-1b-pt \\
                 mistralai/Mistral-Nemo-Base-2407 deepseek-ai/DeepSeek-R1 \\
                 sarvamai/sarvam-30b \\
        --output cross_model_results.json

The default model set in the paper was: Llama-3.2-1B, Qwen3-8B, Gemma-3-1B,
Mistral-Nemo (Tekken), DeepSeek-R1, Sarvam-30B — pass via ``--models``.

This probe does NOT require trained nexusEmbedding checkpoints; it
operates on each model's published input-embedding table directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F


# Default orthographic probe pairs (anchor, near-neighbor by 1-2 char edits).
ORTHO_PAIRS: List[Tuple[str, str, str]] = [
    # (category, anchor, near_neighbor)
    ("plural",    "magnet",    "magnets"),
    ("plural",    "horse",     "horses"),
    ("plural",    "river",     "rivers"),
    ("typo",      "doctor",    "docotr"),
    ("typo",      "capital",   "capitla"),
    ("typo",      "weather",   "wether"),
    ("inflect",   "run",       "runs"),
    ("inflect",   "walk",      "walks"),
    ("inflect",   "sing",      "sings"),
    ("case",      "Apple",     "apple"),
    ("case",      "Run",       "run"),
    ("case",      "Banana",    "banana"),
]

# Semantic category clusters.
CATEGORY_CLUSTERS = {
    "animals": ["dog", "cat", "horse", "cow", "sheep", "goat", "rabbit", "pig"],
    "fruits":  ["apple", "banana", "orange", "grape", "pear", "peach", "mango", "kiwi"],
    "colors":  ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black"],
    "vehicles":["car", "truck", "bicycle", "motorcycle", "bus", "train", "boat", "plane"],
    "numbers": ["one", "two", "three", "four", "five", "six", "seven", "eight"],
}


def string_to_embedding(model, tokenizer, text: str, device: str) -> torch.Tensor:
    """
    Get a single ``(D,)`` embedding for ``text``: mean-pool the input-embedding
    rows of its constituent tokens (skip special tokens). Falls back to
    ``model.get_input_embeddings()`` for the lookup table.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        # Empty after special-token strip — fall back to whitespace-prefixed version.
        ids = tokenizer.encode(" " + text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"tokenizer produced no ids for {text!r}")
    table = model.get_input_embeddings().weight  # (V, D)
    rows = table[torch.tensor(ids, device=device)]
    return rows.mean(dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def probe_one_model(model_id: str, device: str) -> dict:
    """Run both probes for a single HF model id. Returns a dict of metrics."""
    from transformers import AutoModel, AutoTokenizer
    print(f"\n=== {model_id} ===")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    print(f"  loaded ({time.time()-t0:.1f}s)")

    # --- Orthographic probe ---
    ortho = []
    with torch.no_grad():
        for cat, anchor, near in ORTHO_PAIRS:
            a = string_to_embedding(model, tok, anchor, device)
            b = string_to_embedding(model, tok, near, device)
            ortho.append({
                "category": cat,
                "anchor": anchor,
                "near_neighbor": near,
                "cosine": cosine(a, b),
            })
    ortho_mean = sum(r["cosine"] for r in ortho) / len(ortho)
    print(f"  ortho mean cosine: {ortho_mean:.4f}  (over {len(ortho)} pairs)")

    # --- Semantic cluster probe ---
    cluster_embeds = {}
    with torch.no_grad():
        for cat, words in CATEGORY_CLUSTERS.items():
            cluster_embeds[cat] = [string_to_embedding(model, tok, w, device) for w in words]

    within = []
    between = []
    cats = list(CATEGORY_CLUSTERS.keys())
    for i, ci in enumerate(cats):
        # within-cluster
        emb = cluster_embeds[ci]
        for a in range(len(emb)):
            for b in range(a + 1, len(emb)):
                within.append(cosine(emb[a], emb[b]))
        # between-cluster (vs all other clusters)
        for cj in cats[i + 1:]:
            for ea in cluster_embeds[ci]:
                for eb in cluster_embeds[cj]:
                    between.append(cosine(ea, eb))

    w_mean = sum(within) / len(within)
    b_mean = sum(between) / len(between)
    print(f"  semantic within-cluster mean cosine:  {w_mean:.4f}  (n={len(within)})")
    print(f"  semantic between-cluster mean cosine: {b_mean:.4f}  (n={len(between)})")
    print(f"  separation ratio (within / between):  {w_mean/max(b_mean,1e-9):.3f}")

    return {
        "model_id": model_id,
        "ortho_pairs": ortho,
        "ortho_mean_cosine": ortho_mean,
        "semantic_within_mean": w_mean,
        "semantic_between_mean": b_mean,
        "semantic_separation": w_mean / max(b_mean, 1e-9),
        "elapsed_s": time.time() - t0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", required=True,
                   help="One or more HF model ids to probe.")
    p.add_argument("--output", default="cross_model_results.json",
                   help="Path to write the JSON output.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"Cross-model embedding-geometry probe.")
    print(f"  models:   {len(args.models)}")
    print(f"  device:   {args.device}")
    print(f"  output:   {args.output}")

    all_results = []
    for mid in args.models:
        try:
            res = probe_one_model(mid, args.device)
            all_results.append(res)
        except Exception as e:
            print(f"!! {mid}: {type(e).__name__}: {str(e)[:200]}")
            all_results.append({"model_id": mid, "error": f"{type(e).__name__}: {e}"})

    with open(args.output, "w") as f:
        json.dump({
            "ortho_pairs_template": ORTHO_PAIRS,
            "category_clusters_template": CATEGORY_CLUSTERS,
            "results": all_results,
        }, f, indent=2)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
