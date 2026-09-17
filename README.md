# Nexus: byte-level structured embeddings, small-scale reproduction

Scaled-down, fully executable pipeline for the five claims in the Nexus
proposal. Every script is standalone, CLI-driven, and writes its artifacts to
`results/claimN/`, so the same repo runs unchanged on Colab T4 and EC2.

## What proves what

| Claim | Script | Needs GPU? | Runtime | Paper analog it produces |
|---|---|---|---|---|
| 1. Trained tables cluster typographically | `claim1_cross_model_probe.py` | no | ~10-15 min | Tables 5 + 6, qualitative `run -> Run, .run` |
| 2. Codec properties hold on paper | `claim2_codec_properties.py` | no | ~2 min | Tables 2 + 3, unit-norm check, (L-k)/L check, Jaccard |
| 3. Controlled training comparison | `prepare_data.py` then `claim3_train.py` (x2 arms) then `claim3_analyze.py` | yes | 2-2.5 h/run T4 | Tables 7 + 8 + 9, loss curves, gap curve, param accounting |
| 4. Typo robustness of trained models | `claim4_robustness.py` | yes | ~5 min | Tables 13 + 14, qualitative pairs |
| 5. Layered probe (mechanism check) | `claim5_layered_probe.py` | yes | ~10-15 min | Table 10, nation neighborhoods |

Claims 1 and 2 need no training and no GPU: run them first, today, on free
Colab CPU. Claims 4 and 5 reuse the claim 3 checkpoints.

## 4. Claim 1 — Trained Embedding Tables May Not Encode What Is Assumed



### 4.1 Hypothesis under test

Does a trained input-embedding table organize tokens primarily by **word relationships** (the implicit justification for spending trainable parameters on it), or by something narrower — such as surface typography?

### 4.2 Method

Four public models were probed: `gpt2` (vocab 50,257, d=768, tied), `SmolLM2-135M` (49,152, 576, tied), `Pythia-160M` (50,277, 768, untied), `Qwen2.5-0.5B` (151,665, 896). For each model, three retrieval spaces were built over the identical vocabulary: the model's trained table, a random Gaussian table of matching shape, and the NibbleNet codec (`dp = 16`) over the same vocabulary. Twenty probes across four morphological families (*run, compute, magnet, tion*) were queried by mean-centered cosine, top-5, with self-retrieval excluded.

**Metric — loose morph@5:** the fraction of a probe's top-5 retrievals whose *canonical form* (whitespace markers stripped, edge punctuation stripped, case-folded) differs from the probe's own canonical form. A retrieval of the same word in a different case or with a leading space scores 0 (no escape from typographic clustering); anything else scores 1.

### 4.3 Results

**Loose morph@5, all four probe families:**


| Model        | Trained | Random | NibbleNet codec |
| ------------ | ------- | ------ | --------------- |
| gpt2         | 0.74    | 1.00   | 0.94            |
| SmolLM2-135M | 0.67    | 1.00   | 0.93            |
| Pythia-160M  | 0.73    | 1.00   | 0.93            |
| Qwen2.5-0.5B | 0.52    | 1.00   | 0.90            |


**Restricted to single-token probe families only** (*run*, *tion* — removing a multi-token averaging artifact present in *magnet* and *compute*):


| Model        | Trained (clean subset) | NibbleNet codec (clean subset) |
| ------------ | ---------------------- | ------------------------------ |
| gpt2         | 0.56                   | 0.90                           |
| SmolLM2-135M | 0.54                   | 0.90                           |
| Pythia-160M  | 0.60                   | 0.88                           |
| Qwen2.5-0.5B | 0.22                   | 0.84                           |


**Qualitative neighborhoods for the probe *run*:** gpt2's trained table returns `Ġrun, runs, Run, ĠRun, Ġruns` — five surface variants of the same word. Qwen's trained table returns `Ġrun, Run, _run, ĠRun, .run` — again, five surface variants and nothing else. The codec, on the same vocabularies, returns byte-similar but distinct strings: `runs, ru, runner, rub, rug` (gpt2); `runk, rund, runc, runs, runt` (SmolLM2).

**Anisotropy** (mean-vector norm / raw mean pairwise cosine): gpt2 2.05 / 0.268; SmolLM2 2.18 / 0.447; Pythia 0.05 / 0.004; Qwen 0.18 / 0.154. The two weight-tied tables carry a large shared direction; the untied table (Pythia) is nearly isotropic.

### 4.4 Interpretation

On every model tested, the dominant organizing structure of a trained embedding table is **typographic identity**, not word relationships. On the clean, single-token probe subset, roughly half of a trained table's nearest neighbors are the probe word again in a different case or with different leading whitespace; for Qwen2.5-0.5B this rises to about four in five. The deterministic codec escapes this clustering substantially (0.84–0.94 vs. 0.22–0.60). This finding is the empirical premise the rest of the investigation rests on: if the thing a learned table is assumed to buy (semantic organization) is largely typographic clustering instead, a fixed rule that avoids that clustering is a legitimate candidate to test.

### 4.5 Limitations

Four models only, all 124M–500M parameters; nothing larger was downloadable within RAM/gating constraints. The loose metric measures *escape from typographic clustering only* — it does not check whether the escaped neighbors are useful relatives (the codec's `rub`/`rug` count identically to a hypothetically better `runner`/`running`). Multi-token probes inflate the trained-table score, which is why the single-token subset is the primary number. Twenty probes, English only.

---



## 5. Claim 2 — A Fixed Byte-Position Rule Produces Its Own Kind of Structure



### 5.1 Hypothesis under test

Do the two properties derived on paper for the codec — unit norm, and the closed-form same-length cosine `(L−k)/L` — actually hold on real tokenizer vocabularies, and what does the encoding cluster together in practice?

### 5.2 Method

The codec was implemented exactly as defined (one nonzero entry per byte at index `byte × dp + position`, magnitude `1/√L`, `D = 256 × dp`) and subjected to seven checks with no training involved: unit-norm verification across the full GPT-2 vocabulary at `dp = 16`; closed-form cosine on seven hand-picked word pairs; cosine between `run` and `RUN`; top-10 codec neighbors for eight probes; byte-length coverage of four vocabularies at `dp = 16, 32, 64`; single-vector retrieval of out-of-vocabulary strings; cross-tokenizer Jaccard overlap of top-5 neighborhoods across four independently built tokenizers.

### 5.3 Results

**Unit norm** over all 50,257 GPT-2 tokens: min `0.99999994`, max `1.00000012` — holds for every token to numerical precision.

**Closed-form cosine**, exact on every same-length pair tested:


| Pair                | Edit type               | Measured | (L−k)/L |
| ------------------- | ----------------------- | -------- | ------- |
| mistake / mistkae   | transposition           | 0.714    | 0.714   |
| receive / recieve   | transposition           | 0.714    | 0.714   |
| separate / seperate | substitution            | 0.875    | 0.875   |
| realize / realise   | substitution            | 0.857    | 0.857   |
| color / colour      | insertion               | 0.730    | n/a     |
| compute / commute   | unrelated, byte-similar | 0.857    | 0.857   |
| nation / notion     | unrelated, byte-similar | 0.833    | 0.833   |


`cos(run, RUN) = 0.0` exactly (case is encoded as a distinct byte value at every position, so the two strings share no active codec dimensions).

**Neighborhoods (GPT-2 vocabulary):** `run → runs, ru, runner, rub, rug, rum`; `compute → computer, Computer, comp, compl, com, component`; `nation → national, Nation, vation, lation, cation, uation`; `running → Running, funding, ounding, ouncing, ranking, ranging`; `magnetic → mag, kinetic, Genetic, genetic, market, Genetics`.

**Byte-length coverage** (fraction of vocabulary tokens at most `dp` bytes long):


| Vocabulary | dp = 16 | dp = 32 | dp = 64 |
| ---------- | ------- | ------- | ------- |
| gpt2       | 99.86%  | 99.97%  | 99.99%  |
| SmolLM2    | 99.68%  | 99.90%  | 99.99%  |
| Pythia     | 99.46%  | 99.74%  | 99.92%  |
| Qwen       | 98.73%  | 99.84%  | 99.92%  |


**Out-of-vocabulary single-vector retrieval:** `asynchronously → synchron, synchronized, synchronization, Async, sync`; `deserialization → specialization, initialization`; `kubernetes → Internet, internet, uber, Tube`; `shoggoth → thought, show, shop`; `nibbletron → inflation, isolation, abolition`; `tiramisu → thritis, tera, ti`.

**Cross-tokenizer Jaccard** (top-5, mean over six tokenizer pairs): `nation` 0.71, `magnet` 0.59, `station` 0.58, `running` 0.56, `run` 0.44, `compute` 0.43, `computer` 0.37, `magnetic` 0.29 — mean 0.50.

### 5.4 Interpretation

Both on-paper properties hold **exactly** on real vocabularies. The encoding's organizing principle is byte-level locality: two strings are close if and only if they share bytes at the same positions. This produces predictable behavior in both directions — a single-character typo gets a cosine of 0.86–0.88 with its correct spelling, case variants get cosine exactly 0, and neighborhoods are roughly half-shared across four independently built tokenizers (mean Jaccard 0.50) — but also predictable costs: unrelated words at small edit distance score as similar as genuine typos (`compute`/`commute` = 0.857), rhymes cluster with unrelated inflections (`running → funding, ranking`), and insertions are penalized more heavily than substitutions (`color`/`colour` = 0.730). Coverage at `dp = 16` (`D = 4096`) exceeds 98.7% on every vocabulary tested, which is the basis for using `dp = 16` in Claim 3.

### 5.5 Limitations

English-centric tokenizers and English probes only; no non-Latin, emoji, or combining-character strings were tested. The closed-form check is exact by construction for same-length pairs; there is no equivalent formula for insertions, and only one measured example is given. Neighborhood quality is judged by inspection, not by an external gold standard. OOV retrieval is a capability demonstration on seven strings, not a systematic evaluation. None of this establishes whether the encoding *helps* a language model — that is the subject of Claim 3.

---




## Design choices and why 

**Claim 1 models: gpt2, SmolLM2-135M, pythia-160m, Qwen2.5-0.5B.**
The paper used six models up to 671B; downloading DeepSeek-V3's embedding
shard alone is multi-GB and needs the safetensors index trick. These four are
all ungated, under ~1 GB each, and still preserve the *structure* of the
paper's argument: tied models (gpt2, SmolLM2, Qwen2.5-0.5B), one untied model
(pythia-160m), and three byte-level BPE tokenizer variants (GPT-2, GPT-NeoX,
Qwen). SmolLM2-135M is shared with the paper's own set, giving one direct
point of comparison.

**Claim 3 dataset: FineWeb-Edu (sample-10BT, streamed).**
Same dataset family the paper trained on, so the reproduction is a faithful
scale-down, not a different experiment. Streaming pulls only the ~60M tokens
you need. Fallback `--dataset wikitext` if HF streaming is slow.

**Claim 3 model: 6 layers, 6 heads, d=384, ctx 256, GPT-2 vocab (≈30M
total).** nanoGPT's GPT-2 recipe scaled down, keeping the vocab at 50257 on
purpose: the tied table is 19.3M params against a 10.6M body, so we stay in
the input-embedding-dominated regime where the proposal's question is live.
NibbleNet arm: dp=16 (D=4096) because claim 2's coverage table shows dp=16 covers
>99% of GPT-2's English vocab; projection is 4096x384 = 1.6M (a 92% input-side
cut), with the mandatory untied 19.3M head reported honestly in
`param_accounting.csv` exactly as the proposal promises.

**Budget: 3000 steps x 16,384 tok/step ≈ 49M tokens.** The paper used 2.5B
tokens on 124M params; this is the same tokens-per-parameter ballpark at
1/50th the compute. Enough for val loss to be well past the noise floor and
for the gap trajectory (paper Fig/Table 7 shape) to be visible.

**Honesty note.** At this scale the paper's exact 2.5% gap is not guaranteed;
seeds are noisier and the model is 4x smaller. What you are proving is the
methodology plus the direction. Report whatever `claim3_analyze.py` prints,
including "favorable cells" and SNR: a smaller-but-consistent gap is a
legitimate result, and so is a null one honestly reported.

## Option A: Google Colab T4 (free)

Runtime > Change runtime type > T4 GPU.

```bash
# cell 1: setup
!git clone https://github.com/RishitLaddha/nexus.git || true
%cd nexus
!pip -q install -r requirements.txt

# cell 2: claims 1 and 2 (no training)
!python claim1_cross_model_probe.py
!python claim2_codec_properties.py

# cell 3: data (once, ~20 min; copy data/ to Drive so you never redo it)
!python prepare_data.py --train_tokens 60000000 --val_tokens 2000000

# cell 4: claim 3, one run per session to dodge the 12h/idle limits
!python claim3_train.py --arm bpe  --seed 1337
# next session:
!python claim3_train.py --arm nibble --seed 1337
# optional extra seeds when you have sessions to spare:
#   --seed 42 for both arms

# cell 5: analysis + claims 4 and 5
!python claim3_analyze.py
!python claim4_robustness.py --seed 1337
!python claim5_layered_probe.py --seed 1337
```

Colab survival tips: mount Drive and `cp -r runs data /content/drive/...`
after each run; `log.json` is written every eval so a dead session loses at
most 100 steps of information (checkpoint is end-of-run only, so treat a
killed training run as a rerun).

## Option B: EC2 (faster, still cheap)

Do NOT use t2.micro/t3.micro: no GPU, training would take days. GPU options:

| Instance | GPU | On-demand | Spot (typ.) | Per claim-3 run | 2-run total |
|---|---|---|---|---|---|
| g4dn.xlarge | T4 16GB | ~$0.53/h | ~$0.16-0.21/h | ~2-2.5 h | ~$2 on-demand |
| g5.xlarge | A10G 24GB | ~$1.01/h | ~$0.30-0.45/h | ~45-60 min | ~$2 on-demand, <$1 spot |

g5.xlarge spot is the sweet spot: ~3x faster than the free T4, whole claim-3
phase (2 arms) in under 2 hours, a couple of dollars total. Prices vary by
region; check yours.

```bash
# 1. Launch: g5.xlarge, AMI "Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04)",
#    100 GB gp3 root volume, allow SSH from your IP. Spot if available.
ssh -i key.pem ubuntu@<ip>

# 2. Setup (PyTorch is preinstalled on the DLAMI)
git clone https://github.com/RishitLaddha/nexus.git && cd nexus
pip install -r requirements.txt
nvidia-smi   # sanity: A10G visible

# 3. Same commands, back to back inside tmux so SSH drops don't kill runs
tmux new -s nexus
python claim1_cross_model_probe.py
python claim2_codec_properties.py
python prepare_data.py
python claim3_train.py --arm bpe  --seed 1337
python claim3_train.py --arm nibble --seed 1337
python claim3_train.py --arm bpe  --seed 42     # optional 2nd seed
python claim3_train.py --arm nibble --seed 42
python claim3_analyze.py
python claim4_robustness.py --seed 1337
python claim5_layered_probe.py --seed 1337

# 4. Pull results to your laptop, then TERMINATE the instance
scp -r -i key.pem ubuntu@<ip>:~/nexus/results ubuntu@<ip>:~/nexus/runs ./
```

Terminate, don't stop: a stopped g5 still bills the EBS volume.

## Repo layout

```
nexus/codec.py            byte-position codec + NibbleNetEmbedding (gpu_dynamic)
nexus/model.py            minimal GPT, arm = bpe-tied | nibble (only diff)
nexus/probe_utils.py      canonical form, centered cosine, morph@K, families
claim1_cross_model_probe.py
claim2_codec_properties.py
prepare_data.py
claim3_train.py           one arm, one seed per invocation
claim3_analyze.py         tables + plots from runs/*/log.json
claim4_robustness.py
claim5_layered_probe.py
```

