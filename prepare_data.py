"""
Data prep for Claim 3 - streams FineWeb-Edu (same dataset family as the
paper, sample-10BT config) and writes nanoGPT-style uint16 memmap files:
    data/train.bin   (default 60M GPT-2 tokens)
    data/val.bin     (default 2M tokens, held out)

Why FineWeb-Edu and not TinyStories/WikiText:
  * It is literally the paper's training distribution, so the reproduction
    stays a faithful scale-down rather than a different experiment.
  * Streaming means you never download the full 10BT sample; you pull only
    the ~60M tokens you need (~a few hundred MB of text over the wire).
Fallback: --dataset wikitext (wikitext-103-raw-v1) if HF streaming is slow
on your connection; direction of the experiment is unchanged.

Runtime: ~15-25 min on Colab CPU (tokenization-bound). Run once, reuse for
all arms and seeds.
"""

import argparse, os
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("gpt2")
    eot = tok.eos_token_id  # 50256, doc separator like nanoGPT

    if args.dataset == "fineweb":
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
    else:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                          split="train", streaming=True)

    total = args.train_tokens + args.val_tokens
    buf = np.empty(total, dtype=np.uint16)
    filled = 0
    batch, BATCH_DOCS = [], 256
    for ex in ds:
        t = ex["text"]
        if not t or not t.strip():
            continue
        batch.append(t)
        if len(batch) == BATCH_DOCS:
            enc = tok(batch, add_special_tokens=False)["input_ids"]
            for ids in enc:
                ids.append(eot)
                n = min(len(ids), total - filled)
                buf[filled:filled + n] = np.asarray(ids[:n], dtype=np.uint16)
                filled += n
                if filled >= total:
                    break
            batch = []
            print(f"\r tokens: {filled/1e6:.1f}M / {total/1e6:.1f}M",
                  end="", flush=True)
            if filled >= total:
                break
    print()
    assert filled >= total, "stream ended early; lower --train_tokens"

    buf[: args.train_tokens].tofile(f"{args.outdir}/train.bin")
    buf[args.train_tokens: total].tofile(f"{args.outdir}/val.bin")
    print(f"wrote {args.outdir}/train.bin ({args.train_tokens/1e6:.0f}M) and "
          f"val.bin ({args.val_tokens/1e6:.0f}M)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--dataset", choices=["fineweb", "wikitext"],
                    default="fineweb")
    ap.add_argument("--train_tokens", type=int, default=60_000_000)
    ap.add_argument("--val_tokens", type=int, default=2_000_000)
    main(ap.parse_args())
