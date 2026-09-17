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

## Design choices and why (defend these in your report)

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

## If a claim-3 run is too slow on your session

Knobs, in order of preference: `--steps 2000` (still shows the gap
trajectory), `--micro_bs 8 --grad_accum 8` (same tokens/step, less VRAM),
`--n_embd 256 --n_head 4` (smaller model, faster, still input-dominated).
Keep BOTH arms on identical settings, always: the comparison is the result.
