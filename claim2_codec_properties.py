"""
CLAIM 2 - the fixed byte-position rule produces its own kind of structure,
and its two on-paper properties hold exactly.

No training, no GPU, runs in under a minute. Verifies:
  (1) Unit norm: ||kappa(b)|| == 1 for EVERY token in the GPT-2 vocab
      (the sqrt(L) normalization argument from the proposal).
  (2) Closed-form edit prediction: cos = (L-k)/L for same-length pairs,
      checked against the actual codec on the paper's Table 2 pairs.
  (3) Case sensitivity: cos(run, RUN) ~ 0 (byte-level locality).
  (4) Codec nearest neighbors over the GPT-2 vocab (byte-similar mixing).
  (5) dp coverage table (paper Table 3 analog) on available tokenizers.
  (6) OOV-as-single-token retrieval (kubernetes, shoggoth, ...).
  (7) Cross-tokenizer Jaccard stability of codec neighborhoods.

Outputs in results/claim2/: codec_cosine_table.csv, dp_coverage.csv,
neighbors.json, jaccard.csv, checks.json
"""

import argparse, csv, itertools, json, math, os
import torch
from transformers import AutoTokenizer

from nexus.codec import (build_byte_buffer, codec_from_bytes,
                         NibbleNetEmbedding, token_to_bytes)
from nexus.probe_utils import canonical_form, centered_neighbors

TOKENIZERS = ["gpt2", "HuggingFaceTB/SmolLM2-135M",
              "EleutherAI/pythia-160m", "Qwen/Qwen2.5-0.5B"]

EDIT_PAIRS = [  # (probe, variant, type) - paper Table 2
    ("mistake", "mistkae", "transposition (same L)"),
    ("receive", "recieve", "transposition (same L)"),
    ("separate", "seperate", "1-byte substitution"),
    ("realize", "realise", "1-byte substitution"),
    ("color", "colour", "insertion (length change)"),
    ("compute", "commute", "byte-similar non-relative"),
    ("nation", "notion", "byte-similar non-relative"),
]

OOV_PROBES = ["kubernetes", "tensorflow", "asynchronously",
              "deserialization", "shoggoth", "nibbletron", "tiramisu"]

NEIGHBOR_PROBES = ["run", "compute", "nation", "magnet",
                   "computer", "running", "station", "magnetic"]


def raw_cos(a: str, b: str, dp=16):
    v = codec_from_bytes([a.encode(), b.encode()], dp, znorm=False)
    return torch.nn.functional.cosine_similarity(v[0:1], v[1:2]).item()


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    checks = {}

    # ---- (1) unit norm across the whole GPT-2 vocab ----
    tok = AutoTokenizer.from_pretrained("gpt2")
    dp = 16
    tb, tl = build_byte_buffer(tok, dp)
    V = tb.shape[0]
    # raw codec (no z-norm) norms
    norms = []
    for s in range(0, V, 8192):
        ids = torch.arange(s, min(s + 8192, V))
        b, L = tb[ids].long(), tl[ids].long()
        pos = torch.arange(dp)
        mask = pos < L.unsqueeze(-1)
        lin = (b * dp + pos) * mask
        vals = mask.float() / torch.sqrt(L.float()).unsqueeze(-1)
        k = torch.zeros(len(ids), 256 * dp)
        k.scatter_add_(-1, lin, vals)
        norms.append(k.norm(dim=-1))
    norms = torch.cat(norms)
    checks["unit_norm"] = {
        "min": norms.min().item(), "max": norms.max().item(),
        "n_tokens": int(V),
        "pass": bool((norms - 1).abs().max() < 1e-4)}
    print(f"[1] unit norm over {V} GPT-2 tokens: "
          f"min={norms.min():.6f} max={norms.max():.6f}  "
          f"PASS={checks['unit_norm']['pass']}")

    # ---- (2) closed-form (L-k)/L vs measured codec cosine ----
    rows = []
    for a, b, typ in EDIT_PAIRS:
        c = raw_cos(a, b, dp)
        if len(a) == len(b):
            k = sum(x != y for x, y in zip(a.encode(), b.encode()))
            pred = (len(a.encode()) - k) / len(a.encode())
        else:
            pred = float("nan")  # length change: no same-L formula, report measured
        rows.append([a, b, typ, round(c, 3),
                     "" if math.isnan(pred) else round(pred, 3)])
        print(f"[2] {a:9s}/{b:9s} cos={c:.3f} predicted={pred if not math.isnan(pred) else '-'}")
    with open(f"{outdir}/codec_cosine_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe", "variant", "type", "measured_cos", "closed_form_(L-k)/L"])
        w.writerows(rows)
    same_len_ok = all(abs(r[3] - r[4]) < 1e-3 for r in rows if r[4] != "")
    checks["closed_form_match"] = same_len_ok
    print(f"[2] closed-form matches measured on all same-length pairs: {same_len_ok}")

    # ---- (3) case sensitivity ----
    checks["cos_run_RUN"] = round(raw_cos("run", "RUN", dp), 4)
    print(f"[3] cos(run, RUN) = {checks['cos_run_RUN']} (byte-level locality)")

    # ---- (4)+(6) codec neighborhoods over GPT-2 vocab ----
    ke = NibbleNetEmbedding(tb, tl, d_model=8, dp=dp)
    Ek = torch.cat([ke.codec(torch.arange(s, min(s + 4096, V)))
                    for s in range(0, V, 4096)])
    id2tok = {i: t for t, i in tok.get_vocab().items()}
    neigh = {}
    for p in NEIGHBOR_PROBES + OOV_PROBES:
        qv = codec_from_bytes([p.encode()], dp, znorm=True)
        ids = tok.encode(p, add_special_tokens=False)
        qid = ids[0] if len(ids) == 1 else None
        idxs, vals = centered_neighbors(Ek, [qid], K=10, query_vecs=qv)
        neigh[p] = [id2tok.get(int(i), "?") for i in idxs[0]]
        print(f"[4/6] {p:16s} -> {neigh[p][:6]}")
    with open(f"{outdir}/neighbors.json", "w") as f:
        json.dump(neigh, f, indent=2, ensure_ascii=False)

    # ---- (5) dp coverage across tokenizers ----
    cov_rows = []
    for name in TOKENIZERS:
        t = AutoTokenizer.from_pretrained(name)
        lens = [len(token_to_bytes(s)) for s in t.get_vocab().keys()]
        n = len(lens)
        row = [name] + [round(100 * sum(l <= d for l in lens) / n, 2)
                        for d in (16, 32, 64)]
        cov_rows.append(row)
        print(f"[5] {name:35s} dp16 {row[1]}%  dp32 {row[2]}%  dp64 {row[3]}%")
    with open(f"{outdir}/dp_coverage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tokenizer", "dp=16 %", "dp=32 %", "dp=64 %"])
        w.writerows(cov_rows)

    # ---- (7) cross-tokenizer Jaccard of codec neighborhoods ----
    tops = {}
    for name in TOKENIZERS:
        t = AutoTokenizer.from_pretrained(name)
        tb2, tl2 = build_byte_buffer(t, dp)
        ke2 = NibbleNetEmbedding(tb2, tl2, d_model=8, dp=dp)
        V2 = tb2.shape[0]
        E2 = torch.cat([ke2.codec(torch.arange(s, min(s + 4096, V2)))
                        for s in range(0, V2, 4096)])
        inv = {i: s for s, i in t.get_vocab().items()}
        tops[name] = {}
        for p in NEIGHBOR_PROBES:
            qv = codec_from_bytes([p.encode()], dp, znorm=True)
            idxs, _ = centered_neighbors(E2, [None], K=5, query_vecs=qv)
            tops[name][p] = {canonical_form(inv.get(int(i), "?"))
                             for i in idxs[0]}
        del E2
    jrows = []
    for p in NEIGHBOR_PROBES:
        js = []
        for a, b in itertools.combinations(TOKENIZERS, 2):
            A, B = tops[a][p], tops[b][p]
            js.append(len(A & B) / max(len(A | B), 1))
        jrows.append([p, round(sum(js) / len(js), 2)])
        print(f"[7] Jaccard({p}) = {jrows[-1][1]}")
    mean_j = round(sum(r[1] for r in jrows) / len(jrows), 2)
    jrows.append(["MEAN", mean_j])
    with open(f"{outdir}/jaccard.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["probe", "mean_jaccard_top5"]); w.writerows(jrows)

    checks["mean_cross_tokenizer_jaccard"] = mean_j
    with open(f"{outdir}/checks.json", "w") as f:
        json.dump(checks, f, indent=2)
    print(f"\nDone. Artifacts in {outdir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/claim2")
    main(ap.parse_args().outdir)
