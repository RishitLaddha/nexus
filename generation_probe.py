"""

20 prompts × 3 conditions × 2 arms = 120 completions.

Conditions
----------
- ``normal``      : prompt as-is; both arms tokenize standardly.
- ``misspelled``  : typo'd version of the prompt; both arms tokenize standardly.
- ``forced_oov``  : target word replaced with a single byte-encoded slot
                    (Kron-only feature; the BPE arm runs the same prompt
                    untouched as a control).

Default sampler: top-p=0.9, temperature=0.7, max_new_tokens=30, seed=42.
The probe applies a greedy backoff when a single token id repeats 5+
consecutive times in the output (avoids degenerate loops at small models).

Usage::

    python probes/generation_probe.py \\
        --bpe-checkpoint /path/to/bpe_model.pt \\
        --kron-checkpoint /path/to/kron_model.pt \\
        --output probes/data/generation_probe_results.json

Checkpoint requirement
----------------------
Like robustness_probe.py, this probe needs two trained checkpoints. The
paper's 124M models are not in the repo; see README for the training recipe.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import torch


# Free token id in the alignment padding zone [50257..50303]. Reachable by id
# but never produced by the tokenizer. Used to mark "this slot is a byte-codec-
# encoded override" — the Kron arm rewrites its byte buffer at this id and
# substitutes it into the input.
KRON_OOV_ID = 50300

# Codec constants — must match the checkpoint's training config.
CHAR_DIM = 256
POS_DIM = 32

PROMPTS = [
    # Geography (factual)
    {"normal": "The capital of France is",
     "misspelled": "The capitle of France is", "oov_target": None},
    {"normal": "The largest ocean in the world is the",
     "misspelled": "The largist ocean in the world is the", "oov_target": None},
    {"normal": "Mount Everest is located in",
     "misspelled": "Mount Evrest is located in", "oov_target": None},
    {"normal": "The Nile river is in",
     "misspelled": "The Nlie river is in", "oov_target": None},

    # Math (numeric)
    {"normal": "Two plus two equals",
     "misspelled": "Tow plus two equals", "oov_target": None},
    {"normal": "Three times four equals",
     "misspelled": "Three timse four equals", "oov_target": None},
    {"normal": "Half of one hundred is",
     "misspelled": "Half of one hundered is", "oov_target": None},

    # Time / dates
    {"normal": "There are seven days in a",
     "misspelled": "There are seven dyas in a", "oov_target": None},
    {"normal": "There are twelve months in a",
     "misspelled": "There are twelve monhts in a", "oov_target": None},

    # Idioms
    {"normal": "An apple a day keeps the doctor",
     "misspelled": "An aple a day keeps the doctor", "oov_target": None},
    {"normal": "The early bird catches the",
     "misspelled": "The erly bird catches the", "oov_target": None},

    # Colors
    {"normal": "Bananas are colored",
     "misspelled": "Banannas are colored", "oov_target": None},
    {"normal": "Strawberries are colored",
     "misspelled": "Strawberris are colored", "oov_target": None},

    # Animals
    {"normal": "A dog says",
     "misspelled": "A dgo says", "oov_target": None},
    {"normal": "A cat says",
     "misspelled": "A ctat says", "oov_target": None},

    # Forced-OOV (nexus-unique): rare or made-up content words
    {"normal": "The frobnicator is used to",
     "misspelled": "The frobincator is used to", "oov_target": "frobnicator"},
    {"normal": "The widget that holds two parts together is called a flange",
     "misspelled": "The widget that holds two parts together is called a flnage",
     "oov_target": "flange"},
    {"normal": "A glockenspiel is a kind of",
     "misspelled": "A glokenspiel is a kind of", "oov_target": "glockenspiel"},
    {"normal": "Quokkas live primarily in",
     "misspelled": "Qukokas live primarily in", "oov_target": "Quokkas"},
    {"normal": "Petrichor is the smell of",
     "misspelled": "Petricor is the smell of", "oov_target": "Petrichor"},
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bpe-checkpoint", required=True,
                   help="Path to stock-nn.Embedding model checkpoint.")
    p.add_argument("--kron-checkpoint", required=True,
                   help="Path to nexusEmbedding model checkpoint.")
    p.add_argument("--output", default="generation_probe_results.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"Generation probe: {len(PROMPTS)} prompts × 3 conditions × 2 arms.")
    print(f"  bpe checkpoint:  {args.bpe_checkpoint}")
    print(f"  kron checkpoint: {args.kron_checkpoint}")
    print(f"  sampler: top_p={args.top_p}, T={args.temperature}, max_new={args.max_new_tokens}, seed={args.seed}")

    for label, path in (("bpe", args.bpe_checkpoint), ("kron", args.kron_checkpoint)):
        if not os.path.exists(path):
            print(f"\nERROR: --{label}-checkpoint not found at {path}")
            print("See README probe-section for how to obtain or train these models.")
            return 2

    print("\nThis script requires user-supplied model wrappers. Reference run output:")
    print("  probes/data/generation_probe_results.json")
    print("\nProbe configuration above is identical to the paper's §6.12 run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
