"""
Spelling-robustness probe (paper Section 6.11).

For each (clean, typo) prompt pair, compute four metrics on each arm
(BPE-trained model vs Nexus-trained model):

  1. KL(clean || typo) over the full vocab distribution
  2. cosine(final_hidden_clean, final_hidden_typo)
  3. log P(clean_top1 | clean) - log P(clean_top1 | typo)        — log-prob drop
  4. binary: top1(clean) == top1(typo)                            — stability

110 (clean, typo) pairs across 11 categories (geography, math, time, colors,
idioms, body, animals, counting, syntax, misc, people). The typo is always
a 1-2 character change in one content word.

Usage::

    python probes/robustness_probe.py \\
        --bpe-checkpoint /path/to/bpe_model.pt \\
        --nexus-checkpoint /path/to/nexus_model.pt \\
        --output probes/data/robustness_results.json

Checkpoint requirement
----------------------
This probe requires two trained checkpoints — one with a stock nn.Embedding
("BPE") and one with NexusEmbedding ("Nexus"). The paper's 124M-param
checkpoints are not in this repo (they were lost). To reproduce: train
two matched 124M nanoGPT models — one stock, one Nexus — for the same
number of steps on the same data. Then pass their paths via the CLI args.

The 110-prompt set below is the same one used in the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


PROMPT_PAIRS = [
    # ---- Geography ----
    ("geography", "The capital of France is",         "The capitla of France is"),
    ("geography", "The capital of Italy is",          "The capitla of Italy is"),
    ("geography", "The capital of Germany is",        "The captial of Germany is"),
    ("geography", "The capital of Spain is",          "The capital of Spian is"),
    ("geography", "The capital of Japan is",          "The capital of Jaapn is"),
    ("geography", "The capital of China is",          "The capital of Chnia is"),
    ("geography", "The capital of Russia is",         "The capital of Russai is"),
    ("geography", "The largest country in the world is", "The lragest country in the world is"),
    ("geography", "The longest river in the world is",   "The longset river in the world is"),
    ("geography", "Mount Everest is located in",      "Mount Evrest is located in"),
    # ---- Math ----
    ("math", "Two plus two equals",                   "Tow plus two equals"),
    ("math", "Three plus three equals",               "Three plus tree equals"),
    ("math", "Four plus four equals",                 "Four plus fuor equals"),
    ("math", "Five plus five equals",                 "Five plus fiev equals"),
    ("math", "Ten plus ten equals",                   "Ten plus tne equals"),
    ("math", "One plus one equals",                   "One plus oen equals"),
    ("math", "Two times three equals",                "Two timse three equals"),
    ("math", "Three times four equals",               "Three times fuor equals"),
    ("math", "Ten minus five equals",                 "Ten minsu five equals"),
    ("math", "One hundred minus one equals",          "One hundred minsu one equals"),
    # ---- Time ----
    ("time", "There are seven days in a",             "There are seven dyas in a"),
    ("time", "There are twelve months in a",          "There are twelve monhts in a"),
    ("time", "There are twenty four hours in a",      "There are twenty four hrous in a"),
    ("time", "There are sixty seconds in a",          "There are sixty secodns in a"),
    ("time", "There are sixty minutes in an",         "There are sixty minuets in an"),
    ("time", "There are four seasons in a",           "There are four seasosn in a"),
    ("time", "The first month of the year is",        "The first month of the yaer is"),
    ("time", "The last month of the year is",         "The last month of the yaer is"),
    ("time", "Monday Tuesday",                         "Monady Tuesday"),
    ("time", "January February",                       "Januray February"),
    # ---- Colors ----
    ("colors", "Roses are red, violets are",          "Roses are red, voilets are"),
    ("colors", "The sky on a clear day is",           "The skye on a clear day is"),
    ("colors", "Grass is typically the color",        "Grass is typcally the color"),
    ("colors", "The sun appears yellow during the",   "The sun appaers yellow during the"),
    ("colors", "Snow is normally colored",            "Snwo is normally colored"),
    ("colors", "Bananas are colored",                 "Banannas are colored"),
    ("colors", "Strawberries are colored",            "Strawberris are colored"),
    ("colors", "Coal is colored",                     "Cooal is colored"),
    ("colors", "Fire trucks are usually colored",     "Fire trcuks are usually colored"),
    ("colors", "Frogs are typically colored",         "Frgos are typically colored"),
    # ---- Idioms ----
    ("idiom", "An apple a day keeps the doctor",      "An aple a day keeps the doctor"),
    ("idiom", "The early bird catches the",           "The erly bird catches the"),
    ("idiom", "Practice makes",                        "Pracitce makes"),
    ("idiom", "Better late than",                      "Bettr late than"),
    ("idiom", "Time is",                               "Tmie is"),
    ("idiom", "Knowledge is",                          "Knowldge is"),
    ("idiom", "Curiosity killed the",                  "Curiousity killed the"),
    ("idiom", "Every cloud has a silver",              "Every cloud has a silvr"),
    ("idiom", "A picture is worth a thousand",         "A picutre is worth a thousand"),
    ("idiom", "Don't judge a book by its",             "Don't judge a boook by its"),
    # ---- Body ----
    ("body", "Humans have two",                        "Humns have two"),
    ("body", "We see with our",                         "We see wtih our"),
    ("body", "We smell with our",                       "We smel with our"),
    ("body", "We hear with our",                        "We haer with our"),
    ("body", "We taste with our",                       "We tatse with our"),
    ("body", "The human body has two",                  "The humn body has two"),
    ("body", "The heart pumps",                         "The heart pmups"),
    ("body", "The brain is in the",                     "The brian is in the"),
    ("body", "Lungs are used for",                      "Lnugs are used for"),
    ("body", "Teeth are used to",                       "Teath are used to"),
    # ---- Animals ----
    ("animals", "A dog says",                           "A dgo says"),
    ("animals", "A cat says",                           "A ctat says"),
    ("animals", "A cow says",                           "A cwo says"),
    ("animals", "Birds can",                            "Brids can"),
    ("animals", "Fish live in",                         "Fihs live in"),
    ("animals", "Snakes are",                           "Snaks are"),
    ("animals", "Bees produce",                         "Bess produce"),
    ("animals", "Sheep are covered in",                 "Sheap are covered in"),
    ("animals", "Cats have four",                       "Ctas have four"),
    ("animals", "Penguins live in",                     "Pengiuns live in"),
    # ---- Counting ----
    ("counting", "One, two, three,",                    "One, two, tree,"),
    ("counting", "First, second, third,",               "Frist, second, third,"),
    ("counting", "A, B, C,",                            "A, B, C ,"),
    ("counting", "Twenty plus thirty equals",           "Twentty plus thirty equals"),
    ("counting", "Five hundred plus five hundred is",   "Five hundrd plus five hundred is"),
    ("counting", "Half of ten is",                      "Hlaf of ten is"),
    ("counting", "Half of one hundred is",              "Hlaf of one hundred is"),
    ("counting", "Double of five is",                   "Doube of five is"),
    ("counting", "A dozen is",                           "A dozn is"),
    ("counting", "A century is",                         "A centruy is"),
    # ---- Syntax ----
    ("syntax", "She walked to the",                     "She wlaked to the"),
    ("syntax", "He opened the",                         "He opend the"),
    ("syntax", "They went to the",                      "They wnet to the"),
    ("syntax", "The boy and the",                       "The byo and the"),
    ("syntax", "The book is on the",                    "The boook is on the"),
    ("syntax", "He drank a glass of",                   "He drnak a glass of"),
    ("syntax", "She ate a piece of",                    "She aet a piece of"),
    ("syntax", "Children love to",                      "Childern love to"),
    ("syntax", "Birds love to",                         "Brids love to"),
    ("syntax", "Fish love to",                          "Fihs love to"),
    # ---- Misc ----
    ("misc", "Water boils at one hundred",              "Watter boils at one hundred"),
    ("misc", "Water freezes at zero",                   "Watter freezes at zero"),
    ("misc", "The sun rises in the",                    "The sun rises in the"),
    ("misc", "The sun sets in the",                     "The sun stes in the"),
    ("misc", "Plants need sunlight and",                "Plnats need sunlight and"),
    ("misc", "Books are made of",                       "Boks are made of"),
    ("misc", "Bread is made from",                      "Bred is made from"),
    ("misc", "Cheese is made from",                     "Cheees is made from"),
    ("misc", "Milk comes from",                         "Mlik comes from"),
    ("misc", "Eggs come from",                          "Egs come from"),
    # ---- People & professions ----
    ("people", "A doctor treats",                       "A docotr treats"),
    ("people", "A teacher teaches",                     "A teachr teaches"),
    ("people", "A farmer grows",                        "A farmr grows"),
    ("people", "A chef cooks",                          "A chf cooks"),
    ("people", "A pilot flies",                         "A piolt flies"),
    ("people", "A nurse helps",                         "A nrse helps"),
    ("people", "A scientist studies",                   "A scientst studies"),
    ("people", "An artist paints",                      "An artst paints"),
    ("people", "A baker bakes",                         "A bakr bakes"),
    ("people", "A driver drives a",                     "A drvier drives a"),
]


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(p || q) over softmaxed distributions, in nats."""
    p_log = torch.log_softmax(p_logits.float(), dim=-1)
    q_log = torch.log_softmax(q_logits.float(), dim=-1)
    p = p_log.exp()
    return (p * (p_log - q_log)).sum().item()


def run_pair(model, tokenizer, clean: str, typo: str, device: str) -> dict:
    """
    Run one (clean, typo) pair through ``model`` and compute the 4 metrics.

    ``model`` must satisfy: callable with input_ids (B, L), returns
    (logits (B, L, V), final_hidden (B, L, D)). The probe uses position L-1
    of each as the "next-token" prediction.
    """
    raise NotImplementedError(
        "Plug your specific model/tokenizer here. See the README probe-section "
        "for an example wrapper. The 110-prompt set and metric definitions in "
        "this file are stable; only the model-loading bit needs adapting."
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bpe-checkpoint", required=True,
                   help="Path to stock-nn.Embedding model checkpoint.")
    p.add_argument("--kron-checkpoint", required=True,
                   help="Path to KroneckerEmbedding model checkpoint.")
    p.add_argument("--output", default="robustness_results.json",
                   help="Where to write the JSON result.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"Probe: spelling robustness, {len(PROMPT_PAIRS)} prompt pairs.")
    print(f"  bpe checkpoint:  {args.bpe_checkpoint}")
    print(f"  kron checkpoint: {args.kron_checkpoint}")
    print(f"  device:          {args.device}")

    # Verify checkpoints exist before proceeding.
    for label, path in (("bpe", args.bpe_checkpoint), ("kron", args.kron_checkpoint)):
        if not os.path.exists(path):
            print(f"\nERROR: --{label}-checkpoint not found at {path}")
            print("See the README probe-section for how to obtain or train these models.")
            return 2

    print("\nThis script requires user-supplied model wrappers (see run_pair stub).")
    print("Reference output schema and the original 110-prompt run are in:")
    print("  probes/data/robustness_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
