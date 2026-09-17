"""
CLAIM 4 - the codec's closed-form edit prediction should show up as
observable model behavior on the TRAINED checkpoints.

40 (clean, typo) prompt pairs across 8 categories, 5 per category (a scaled
version of the paper's 110-pair probe). Each typo is a single substitution
or transposition in one content word. Four metrics on the next-token
distribution, exactly as in the paper's Section 6.11:

  top-1 match rate (higher better)      mean KL(clean||typo) (lower better)
  final hidden-state cosine (higher)    mean delta log p (lower better)

Run AFTER claim3 training:
    python claim4_robustness.py --seed 1337

Outputs: results/claim4/aggregate.csv, per_category.csv, qualitative.json
Runtime: ~5 min on T4.
"""

import argparse, csv, json, os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from nexus.model import GPT, GPTConfig
from nexus.codec import build_byte_buffer

PAIRS = {
    "animals": [
        ("The lion is the king of the", "The loin is the king of the"),
        ("A dog wagging its", "A dog waggnig its"),
        ("Birds fly south for the", "Birds fly south for teh"),
        ("The cat chased the", "The cat chsed the"),
        ("An elephant never", "An elehpant never"),
    ],
    "body": [
        ("The heart pumps", "The heart pmups"),
        ("We see with our", "We see wtih our"),
        ("The brain controls the", "The brian controls the"),
        ("Lungs are used for", "Lnugs are used for"),
        ("Blood flows through the", "Blood folws through the"),
    ],
    "colors": [
        ("Roses are red, violets are", "Roses are red, voilets are"),
        ("The sky is", "The sky si"),
        ("Grass is usually", "Grass is usaully"),
        ("Snow is white and coal is", "Snow is whtie and coal is"),
        ("The color of the sun is", "The colro of the sun is"),
    ],
    "counting": [
        ("One two three four", "One two thre four"),
        ("A week has seven", "A week has sevn"),
        ("A dozen equals", "A dozne equals"),
        ("Ten minus one is", "Ten mnius one is"),
        ("A pair means", "A piar means"),
    ],
    "geography": [
        ("The capital of France is", "The capitla of France is"),
        ("Mount Everest is the tallest", "Mount Evrest is the tallest"),
        ("The Nile is a famous", "The Nlie is a famous"),
        ("Oceans are full of salt", "Ocaens are full of salt"),
        ("Deserts are very", "Dserts are very"),
    ],
    "idiom": [
        ("Practice makes", "Pracitce makes"),
        ("Knowledge is", "Knowldge is"),
        ("Actions speak louder than", "Actions speak luoder than"),
        ("Better late than", "Better late thna"),
        ("Every cloud has a silver", "Every cloud has a silvr"),
    ],
    "syntax": [
        ("She was walking to the", "She was walknig to the"),
        ("They have been working on", "They have been wroking on"),
        ("He quickly ran towards", "He quikcly ran towards"),
        ("The book was written by", "The book was writen by"),
        ("It is important to", "It is improtant to"),
    ],
    "time": [
        ("The sun rises in the", "The sun rsies in the"),
        ("Twelve o'clock at night is", "Twelve o'clcok at night is"),
        ("Yesterday comes before", "Yestreday comes before"),
        ("A year has twelve", "A year has twelev"),
        ("Winter comes after", "Wintre comes after"),
    ],
}


def load_arm(arm, seed, device, runs="runs"):
    ck = torch.load(f"{runs}/{arm}_s{seed}/ckpt.pt", map_location=device)
    a = ck["config"]
    cfg = GPTConfig(vocab_size=50257, block_size=a["block"],
                    n_layer=a["n_layer"], n_head=a["n_head"],
                    n_embd=a["n_embd"], embedding=arm, dp=a["dp"])
    if arm == "nibble":
        tok = AutoTokenizer.from_pretrained("gpt2")
        tb, tl = build_byte_buffer(tok, a["dp"])
        m = GPT(cfg, tb, tl)
    else:
        m = GPT(cfg)
    m.load_state_dict(ck["model"])
    return m.to(device).eval()


@torch.no_grad()
def probe_pair(model, tok, clean, typo, device):
    out = {}
    hs = {}
    for name, text in (("clean", clean), ("typo", typo)):
        ids = torch.tensor([tok.encode(text)], device=device)
        logits, _, hid = model(ids, return_hidden=True)
        out[name] = F.log_softmax(logits[0, -1].float(), dim=-1)
        hs[name] = hid[-1][0, -1].float()
    pc, pt = out["clean"], out["typo"]
    top1c, top1t = pc.argmax().item(), pt.argmax().item()
    kl = torch.sum(pc.exp() * (pc - pt)).item()
    cos = F.cosine_similarity(hs["clean"], hs["typo"], dim=0).item()
    dlogp = (pc[top1c] - pt[top1c]).item()
    return {"match": int(top1c == top1t), "kl": kl, "cos": cos,
            "dlogp": dlogp, "top1_clean": tok.decode([top1c]),
            "top1_typo": tok.decode([top1t])}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("gpt2")
    os.makedirs(args.outdir, exist_ok=True)
    models = {a: load_arm(a, args.seed, device) for a in ("bpe", "nibble")}

    rows, qual = [], []
    agg = {a: {"match": [], "kl": [], "cos": [], "dlogp": []}
           for a in ("bpe", "nibble")}
    for cat, pairs in PAIRS.items():
        catres = {a: {"match": [], "kl": []} for a in ("bpe", "nibble")}
        for clean, typo in pairs:
            rec = {"category": cat, "clean": clean, "typo": typo}
            for a in ("bpe", "nibble"):
                r = probe_pair(models[a], tok, clean, typo, device)
                for k in ("match", "kl", "cos", "dlogp"):
                    agg[a][k].append(r[k])
                catres[a]["match"].append(r["match"])
                catres[a]["kl"].append(r["kl"])
                rec[a] = r
            qual.append(rec)
        rows.append([cat] + [round(sum(catres[a]["match"]) / 5, 2)
                             for a in ("bpe", "nibble")]
                    + [round(sum(catres[a]["kl"]) / 5, 2)
                       for a in ("bpe", "nibble")])
        print(f"{cat:10s} top1 bpe {rows[-1][1]:.2f} nibble {rows[-1][2]:.2f}  "
              f"KL bpe {rows[-1][3]:.2f} nibble {rows[-1][4]:.2f}")

    def mean(a, k):
        return sum(agg[a][k]) / len(agg[a][k])

    with open(f"{args.outdir}/aggregate.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "bpe", "nibble", "delta"])
        w.writerow(["top1_match_rate", round(mean("bpe", "match"), 3),
                    round(mean("nibble", "match"), 3),
                    f"{100*(mean('nibble','match')-mean('bpe','match')):+.1f} pp"])
        for k, better in (("kl", "lower"), ("cos", "higher"),
                          ("dlogp", "lower")):
            b, kr = mean("bpe", k), mean("nibble", k)
            w.writerow([f"mean_{k} ({better} better)", round(b, 3),
                        round(kr, 3), f"{100*(kr-b)/abs(b):+.1f}%"])
    with open(f"{args.outdir}/per_category.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "top1_bpe", "top1_nibble", "kl_bpe", "kl_nibble"])
        w.writerows(rows)
    with open(f"{args.outdir}/qualitative.json", "w") as f:
        json.dump(qual, f, indent=2, ensure_ascii=False)

    print(f"\nAGGREGATE  top1: bpe {mean('bpe','match'):.3f} vs nibble "
          f"{mean('nibble','match'):.3f} | KL: {mean('bpe','kl'):.3f} vs "
          f"{mean('nibble','kl'):.3f} | cos: {mean('bpe','cos'):.3f} vs "
          f"{mean('nibble','cos'):.3f}")
    print(f"Done. Artifacts in {args.outdir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--outdir", default="results/claim4")
    main(ap.parse_args())
