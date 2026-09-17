"""
CLAIM 5 - check the obvious mechanism story directly: does the fixed rule
help the model build word-relationship (morphological) structure in the
first transformer layers? (Paper Section 6.8; expect: it does NOT, in
either arm, at small scale. BPE trades structure for context; NibbleNet
preserves byte geometry.)

For every vocab token we compute three representations in each trained arm:
    Eraw : input pathway output (embedding / codec+projection)
    L0   : output of transformer block 0 (position 0, post-residual)
    L1   : output of transformer block 1
and measure loose / root-substring / strict-family morph@10 for the
hand-curated families in nexus/probe_utils.py.

Run AFTER claim3 training:
    python claim5_layered_probe.py --seed 1337

Outputs: results/claim5/layered_table.csv (paper Table 10 analog),
         qualitative_nation.json, layered_plot.png
Runtime: ~10-15 min on T4 (full-vocab forward through 2 blocks, batched).
"""

import argparse, csv, json, os
import torch
from transformers import AutoTokenizer

from claim4_robustness import load_arm
from nexus.probe_utils import (STRICT_FAMILIES, centered_neighbors,
                               loose_morph_at_k, root_morph_at_k,
                               strict_morph_at_k)


@torch.no_grad()
def vocab_reps(model, V, device, bs=2048):
    """[V, d] matrices at Eraw / L0 / L1 (each token as a length-1 sequence
    at position 0, consistent across arms)."""
    reps = {"Eraw": [], "L0": [], "L1": []}
    pos0 = model.wpe(torch.zeros(1, dtype=torch.long, device=device))
    for s in range(0, V, bs):
        ids = torch.arange(s, min(s + bs, V), device=device).unsqueeze(1)
        e = model.wte(ids)                    # [b,1,d]
        reps["Eraw"].append(e[:, 0].float().cpu())
        x = e + pos0
        x = model.h[0](x)
        reps["L0"].append(x[:, 0].float().cpu())
        x = model.h[1](x)
        reps["L1"].append(x[:, 0].float().cpu())
    return {k: torch.cat(v) for k, v in reps.items()}


def probe_layer(E, tok, id2tok, K=10):
    scores = {"L": [], "M": [], "S": []}
    quals = {}
    for fam, spec in STRICT_FAMILIES.items():
        for p in spec["probes"]:
            ids = tok.encode(p, add_special_tokens=False)
            if len(ids) == 1:
                idxs, _ = centered_neighbors(E, [ids[0]], K)
            else:
                qv = E[ids].mean(0, keepdim=True)
                idxs, _ = centered_neighbors(E, [None], K, query_vecs=qv)
            toks = [id2tok.get(int(i), "?") for i in idxs[0]]
            scores["L"].append(loose_morph_at_k(p, toks))
            scores["M"].append(root_morph_at_k(spec["root"], toks))
            scores["S"].append(strict_morph_at_k(set(spec["family"]), toks))
            if p == "nation":
                quals[p] = toks
    return {k: sum(v) / len(v) for k, v in scores.items()}, quals


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("gpt2")
    id2tok = {i: t for t, i in tok.get_vocab().items()}
    os.makedirs(args.outdir, exist_ok=True)

    table, qual = [], {}
    for arm in ("bpe", "nibble"):
        model = load_arm(arm, args.seed, device)
        reps = vocab_reps(model, 50257, device)
        del model
        for layer in ("Eraw", "L0", "L1"):
            sc, q = probe_layer(reps[layer], tok, id2tok)
            table.append([arm, layer, round(sc["L"], 2),
                          round(sc["M"], 2), round(sc["S"], 2)])
            qual[f"{arm}_{layer}_nation"] = q.get("nation", [])
            print(f"{arm:5s} {layer:5s}  loose {sc['L']:.2f}  "
                  f"root {sc['M']:.2f}  strict {sc['S']:.2f}")
        # delta row (L1 - Eraw), the paper's headline number
        e, l1 = table[-3], table[-1]
        table.append([arm, "delta(L1-Eraw)",
                      round(l1[2] - e[2], 2), round(l1[3] - e[3], 2),
                      round(l1[4] - e[4], 2)])

    with open(f"{args.outdir}/layered_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "layer", "loose@10", "root@10", "strict@10"])
        w.writerows(table)
    with open(f"{args.outdir}/qualitative_nation.json", "w") as f:
        json.dump(qual, f, indent=2, ensure_ascii=False)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        layers = ["Eraw", "L0", "L1"]
        for i, arm in enumerate(("bpe", "nibble")):
            s = [r[4] for r in table if r[0] == arm and r[1] in layers]
            plt.plot(layers, s, marker="o",
                     label=f"{arm} strict-family@10")
        plt.ylabel("strict morph@10"); plt.legend()
        plt.title("Does early-layer morphology emerge? (claim 5)")
        plt.tight_layout(); plt.savefig(f"{args.outdir}/layered_plot.png",
                                        dpi=150)
    except Exception as e:
        print("plot skipped:", e)
    print(f"Done. Artifacts in {args.outdir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--outdir", default="results/claim5")
    main(ap.parse_args())
