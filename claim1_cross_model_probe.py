"""
CLAIM 1 - trained embedding tables may not encode what's assumed.

Probes the ACTUAL trained input embeddings of 4 small public LMs and compares
three retrieval spaces for the same probe words:
  (a) trained embedding table
  (b) random Gaussian table of the same shape (chance baseline)
  (c) Nexus byte-position codec over the same tokenizer's vocab

Outputs (in results/claim1/):
  - loose_morph_table.csv      (paper Table 5 analog)
  - anisotropy_table.csv       (paper Table 6 analog)
  - qualitative_neighbors.json (run -> Run, run, .run ... evidence)
  - loose_morph_bar.png

Why THESE models (and not the paper's six):
  * gpt2 (124M, tied, GPT-2 BPE)            ~ 550 MB download
  * HuggingFaceTB/SmolLM2-135M (tied)       ~ 270 MB, shared with the paper
  * EleutherAI/pythia-160m (UNTIED, NeoX)   ~ 375 MB  -> gives an untied point
  * Qwen/Qwen2.5-0.5B (Qwen BBPE family)    ~ 1 GB    -> third tokenizer family
  All four are small, ungated, free, and fit Colab RAM. Together they still
  span tied/untied and three byte-level-BPE tokenizer variants, which is the
  structure of the paper's argument at 1/1000th the download cost.

Runtime: ~10-15 min on Colab (mostly downloads). GPU not required.
"""

import argparse, json, os, csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from nexus.codec import build_byte_buffer, codec_from_bytes, token_to_bytes
from nexus.probe_utils import (PROBE_FAMILIES, centered_neighbors,
                               loose_morph_at_k, anisotropy)

MODELS = [
    "gpt2",
    "HuggingFaceTB/SmolLM2-135M",
    "EleutherAI/pythia-160m",
    "Qwen/Qwen2.5-0.5B",
]

def get_trained_embedding(name):
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    E = model.get_input_embeddings().weight.detach().clone()
    del model
    return E

def probe_space(E, tokenizer, id2tok, K=5, query_from_codec=None):
    """Mean loose morph@K across all probe families for one retrieval space."""
    per_family, quals = {}, {}
    for fam, probes in PROBE_FAMILIES.items():
        scores = []
        for p in probes:
            ids = tokenizer.encode(p, add_special_tokens=False)
            if query_from_codec is not None:
                qv = query_from_codec([p])          # codec query for string p
                qid = ids[0] if len(ids) == 1 else None
                idxs, _ = centered_neighbors(E, [qid], K, query_vecs=qv)
            elif len(ids) == 1:
                idxs, _ = centered_neighbors(E, [ids[0]], K)
            else:
                qv = E[ids].float().mean(0, keepdim=True)
                idxs, _ = centered_neighbors(E, [None], K, query_vecs=qv)
            toks = [id2tok.get(int(i), "?") for i in idxs[0]]
            scores.append(loose_morph_at_k(p, toks))
            if p in ("run", "nation"):
                quals[p] = toks
        per_family[fam] = sum(scores) / len(scores)
    mean = sum(per_family.values()) / len(per_family)
    clean = (per_family["run"] + per_family["tion"]) / 2  # artifact-free pair
    return mean, clean, per_family, quals

def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    rows, aniso_rows, qual = [], [], {}
    for name in MODELS:
        print(f"\n=== {name} ===")
        tok = AutoTokenizer.from_pretrained(name)
        id2tok = {i: t for t, i in tok.get_vocab().items()}
        E = get_trained_embedding(name)
        # some models (e.g. Pythia) pad the embedding matrix beyond the
        # tokenizer's real vocab; drop the padded dummy rows so all three
        # retrieval spaces cover exactly the same token set
        E = E[: max(tok.get_vocab().values()) + 1]
        V, d = E.shape
        print(f"vocab {V}, d_model {d}")

        # (a) trained
        m_tr, c_tr, fam_tr, q_tr = probe_space(E, tok, id2tok)
        # (b) random Gaussian, same shape
        Er = torch.randn(V, d) * 0.02
        m_rd, c_rd, _, _ = probe_space(Er, tok, id2tok)
        # (c) NibbleNet codec over the same vocab (dp=16 is enough for English BPE)
        dp = 16
        tb, tl = build_byte_buffer(tok, dp)
        # build full codec table in chunks (fp32, ~[V,4096])
        from nexus.codec import NibbleNetEmbedding
        ke = NibbleNetEmbedding(tb, tl, d_model=8, dp=dp)  # proj unused
        Ek = torch.cat([ke.codec(torch.arange(s, min(s + 4096, V)))
                        for s in range(0, V, 4096)])
        codec_query = lambda strs: codec_from_bytes(
            [s.encode() for s in strs], dp, znorm=True)
        m_kr, c_kr, fam_kr, q_kr = probe_space(Ek, tok, id2tok,
                                               query_from_codec=codec_query)

        mu, cosbar = anisotropy(E)
        rows.append([name, round(m_tr, 2), round(m_rd, 2), round(m_kr, 2),
                     round(c_tr, 2), round(c_kr, 2)])
        aniso_rows.append([name, round(mu, 2), round(cosbar, 3)])
        qual[name] = {"trained": q_tr, "nibble": q_kr,
                      "per_family_trained": fam_tr, "per_family_nibble": fam_kr}
        del E, Er, Ek

    with open(f"{outdir}/loose_morph_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "trained@5", "random@5", "nibble@5",
                    "trained_clean(run+tion)", "nibble_clean(run+tion)"])
        w.writerows(rows)
    with open(f"{outdir}/anisotropy_table.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "||mu||", "raw_pairwise_cos"])
        w.writerows(aniso_rows)
    with open(f"{outdir}/qualitative_neighbors.json", "w") as f:
        json.dump(qual, f, indent=2, ensure_ascii=False)

    # bar chart
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        names = [r[0].split("/")[-1] for r in rows]
        x = np.arange(len(names)); w = 0.27
        plt.figure(figsize=(8, 4))
        plt.bar(x - w, [r[1] for r in rows], w, label="Trained")
        plt.bar(x,     [r[2] for r in rows], w, label="Random")
        plt.bar(x + w, [r[3] for r in rows], w, label="NibbleNet codec")
        plt.xticks(x, names, rotation=15); plt.ylabel("loose morph@5")
        plt.title("Escape from typographic clustering (higher = escapes)")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{outdir}/loose_morph_bar.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)

    print("\nloose morph@5 (trained / random / nibble):")
    for r in rows:
        print(f"  {r[0]:35s} {r[1]:.2f} / {r[2]:.2f} / {r[3]:.2f}")
    print(f"\nDone. Artifacts in {outdir}/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/claim1")
    main(ap.parse_args().outdir)
