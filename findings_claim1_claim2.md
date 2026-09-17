# Findings: Claim 1 and Claim 2

## Claim 1: what a trained embedding table actually organizes tokens by

### What we did

We took the trained input embedding tables of four public language models and asked what each table's nearest neighbours look like for a set of probe words. The four models were gpt2 (124M, tied embeddings, GPT-2 BPE, vocab 50,257, d = 768), SmolLM2-135M (tied, vocab 49,152, d = 576), Pythia-160M (untied, GPT-NeoX BPE, vocab 50,277, d = 768) and Qwen2.5-0.5B (Qwen byte-level BPE, vocab 151,665, d = 896). This gives three tokenizer variants and both tied and untied output heads.

For every model we built three retrieval spaces over the same vocabulary: (a) the trained table itself, (b) a random Gaussian table of the same shape as a chance baseline, and (c) the NibbleNet byte-position codec applied to that tokenizer's vocab at dp = 16. In each space we ran the same 20 probe words, four families of five: run (run, runs, running, runner, ran), compute (compute, computer, computing, computation, computes), magnet (magnet, magnets, magnetic, magnetize, magnetized) and tion (nation, station, action, rotation, creation). Retrieval is mean-centred cosine, top-5 excluding the probe itself. Single-token probes use their own row as the query; multi-token probes use the mean of their sub-token rows.

The metric is loose morph@5: the fraction of the five retrievals whose canonical form (whitespace markers and edge punctuation stripped, case folded) differs from the probe's own canonical form. A retrieval like run -> Run or run -> .run scores 0 because it is the same word in a different surface form. A retrieval like run -> rund or run -> runner scores 1. So the metric measures how much a space escapes typographic clustering. It does not check whether retrievals are true morphological relatives; that stricter question is deferred to Claim 5 on our own trained checkpoints.

We also recorded two anisotropy numbers per trained table: the L2 norm of the mean embedding vector, and the raw uncentred mean pairwise cosine over 20,000 random token pairs.

### What we found

Loose morph@5, aggregate over all four families:

| Model | Trained | Random | NibbleNet codec |
|---|---|---|---|
| gpt2 | 0.74 | 1.00 | 0.94 |
| SmolLM2-135M | 0.67 | 1.00 | 0.93 |
| Pythia-160M | 0.73 | 1.00 | 0.93 |
| Qwen2.5-0.5B | 0.52 | 1.00 | 0.90 |
| Mean | 0.67 | 1.00 | 0.93 |

Restricted to the two families that are free of the multi-token artifact described below (run + tion):

| Model | Trained (clean) | NibbleNet (clean) |
|---|---|---|
| gpt2 | 0.56 | 0.90 |
| SmolLM2-135M | 0.54 | 0.90 |
| Pythia-160M | 0.60 | 0.88 |
| Qwen2.5-0.5B | 0.22 | 0.84 |
| Mean | 0.48 | 0.88 |

The ordering is the same on every model: trained < codec < random. Trained embeddings escape typographic clustering far less than the codec does, on all four models, across tied and untied heads and three tokenizer variants. On the clean subset, roughly half of a trained table's top-5 neighbours are just the probe word again in a different case or with a leading space or punctuation mark. For Qwen it is closer to four out of five.

The qualitative neighbourhoods make this concrete. For the probe run:

- gpt2 trained: Ġrun, runs, Run, ĠRun, Ġruns
- SmolLM2 trained: Run, Ġrun, ĠRun, Ġruns, running
- Pythia trained: Ġrun, Run, Ġruns, ĠRun, Ġran
- Qwen trained: Ġrun, Run, _run, ĠRun, .run

Qwen's list is five typographic variants of the same word and nothing else, including the code-style _run and .run. The smaller byte-BPE models let one or two inflected forms in (runs, running, ran) but the neighbourhood is still dominated by case and whitespace variants.

The codec on the same probe, same vocab:

- gpt2 codec: runs, ru, runner, rub, rug
- SmolLM2 codec: runk, rund, runc, runs, runt
- Pythia codec: ru, runner, ruz, rub, rug
- Qwen codec: runs, ru, runner, ruz, ruk

These are byte-similar strings. Some are morphological relatives (runs, runner), some are byte-similar non-relatives (rub, rug, runk). The codec never returns a case or whitespace variant of the probe because those share no bytes at the same positions.

For nation, the trained tables return Nation, Ġnation, ĠNation, national, country. Pythia's trained table returns ation, n, Ġn, ations, N, which are sub-word fragments rather than anything word-like. The codec returns national, vation, lation, cation, uation on every model: a byte-similar suffix family.

Per-family detail (from qualitative_neighbors.json) shows one artifact we have to report. The magnet family scores 0.92 to 1.00 on trained tables too, which looks like trained embeddings escaping typographic clustering there. They are not. magnet, magnetize and magnetized are multi-token under all four tokenizers, so the query is a mean of sub-token rows, and the neighbours end up being other tokens near that mean rather than typographic variants of a single piece. The compute family shows a softer version of the same thing (0.64 to 0.88 trained). That is why we report the run + tion subset as the primary number: those probes are single tokens in every tokenizer and the metric is clean there.

Anisotropy of the trained tables:

| Model | Mean-vector norm | Raw pairwise cosine |
|---|---|---|
| gpt2 | 2.05 | 0.268 |
| SmolLM2-135M | 2.18 | 0.447 |
| Pythia-160M | 0.05 | 0.004 |
| Qwen2.5-0.5B | 0.18 | 0.154 |

The two tied models (gpt2, SmolLM2) have a large common mean direction and high raw pairwise cosine: almost every pair of tokens in SmolLM2's table has cosine 0.45 before centring. Pythia, the one untied model, is nearly isotropic (norm 0.05, pairwise 0.004). Qwen sits in between. This is why mean-centring before retrieval is mandatory; without it, in-family and random pairs would look similar on the tied models.

### Conclusion for Claim 1

The assumption that a trained embedding table organizes tokens by word relationships is not supported by what we measured. Across four models the dominant structure at the input layer is typographic identity: the same string in different case, with or without a leading space, with or without adjacent punctuation. On the clean probe subset trained tables score 0.48 on escape from that clustering versus 0.88 for the deterministic codec. That is the result that opens the door to Claim 2: if the learned table is mainly encoding surface identity, a fixed rule built from surface bytes is a candidate replacement worth defining precisely.

### Limitations and why we stopped here

- Four models, all small (124M to 500M). Larger models were out of reach: the biggest vocabularies would need multi-gigabyte embedding downloads and several gigabytes of RAM per probe, beyond the free Colab tier we ran this on, and some are gated. The pattern is consistent across the four we could run, but we cannot claim it from our own data for models above 0.5B.
- The loose metric only measures escape from typographic clustering. It says nothing about whether the codec's byte-similar neighbours are useful morphological relatives; rub and rug count the same as runs and runner. A strict family-membership metric is applied in Claim 5 on our own trained models, where we control the vocabulary.
- Multi-token probes inflate the trained score (the magnet artifact). We flagged and worked around it by reporting the clean subset, but a larger single-token probe set would be a cleaner design.
- 20 probes, four families, English only. The probe set is small enough that a single family can move the aggregate by a few points, which is another reason the clean-subset number is the one to trust.
- Anisotropy is reported as a diagnostic, not explained. We see tied models with large mean vectors and the untied one nearly isotropic, but with one untied model we cannot separate tying from training data or scale as the cause.

## Claim 2: the byte-position rule and its properties

### What we did

We implemented the encoding exactly as defined: for a token with UTF-8 bytes b1 to bL, one nonzero entry per byte at linear index (byte value x dp + position), each equal to 1/sqrt(L), giving a vector of dimension D = 256 x dp. We then checked, on real tokenizer vocabularies rather than toy examples, whether the two properties derived on paper actually hold, and characterized what the encoding clusters.

Seven checks, no training involved:

1. Unit norm over every token in the GPT-2 vocabulary (50,257 tokens) at dp = 16.
2. The closed-form cosine prediction (L - k)/L for same-length pairs differing in k byte positions, against measured codec cosine on seven word pairs.
3. Case sensitivity: cosine between run and RUN.
4. Codec nearest neighbours for eight probe words over the GPT-2 vocab (mean-centred cosine, top-10).
5. Byte-length coverage of four tokenizer vocabularies at dp = 16, 32, 64.
6. Out-of-vocabulary strings encoded as a single vector and retrieved against the vocab.
7. Cross-tokenizer stability: Jaccard overlap of top-5 canonical-form neighbourhoods for the same probe across the four tokenizers.

### What we found

Unit norm. Across all 50,257 GPT-2 tokens the raw codec norm is min 0.99999994, max 1.00000012, i.e. exactly 1 up to float32 rounding. The scale-consistency property holds for every token regardless of length, not just on average.

Closed form. Measured cosine equals (L - k)/L on every same-length pair:

| Probe | Variant | Type | Measured | (L - k)/L |
|---|---|---|---|---|
| mistake | mistkae | transposition | 0.714 | 0.714 |
| receive | recieve | transposition | 0.714 | 0.714 |
| separate | seperate | 1-byte substitution | 0.875 | 0.875 |
| realize | realise | 1-byte substitution | 0.857 | 0.857 |
| color | colour | insertion (length change) | 0.730 | n/a |
| compute | commute | byte-similar non-relative | 0.857 | 0.857 |
| nation | notion | byte-similar non-relative | 0.833 | 0.833 |

The prediction is exact. A single-character typo in a seven or eight letter word leaves the encoding at cosine 0.86 to 0.88 with the correct spelling; a two-letter transposition leaves it at 0.71. The insertion case (color/colour) drops to 0.73 despite being a one-letter change, because every byte after the insertion point shifts to a new position. This is the concrete number Claim 4 tests against trained-model behaviour.

The same table also shows the cost of the property. compute/commute and nation/notion get 0.86 and 0.83 despite being unrelated words. The encoding cannot tell a typo from a near-neighbour word; that disambiguation has to happen downstream.

Case sensitivity. cos(run, RUN) = 0.0 exactly. Upper and lower case share no bytes at any position, so the encoding treats them as unrelated. This is the mirror image of what Claim 1 found in trained tables, where run and Run are among each other's closest neighbours.

Neighbourhoods over GPT-2 vocab (top-6 shown):

- run -> runs, ru, runner, rub, rug, rum
- compute -> computer, Computer, comp, compl, com, component
- nation -> national, Nation, vation, lation, cation, uation
- station -> Station, utation, otation, itation, utations, itations
- running -> Running, funding, ounding, ouncing, ranking, ranging
- magnet -> mag, market, Magn, mage, markets, mann
- magnetic -> mag, kinetic, Genetic, genetic, market, Genetics

Prefix-sharing words retrieve their family (compute -> computer, component). Suffix-heavy probes retrieve by suffix position (station -> utation, itation). Two entries show the encoding's weak spots directly: running -> funding, ranking, ranging are rhymes, not relatives, and magnetic -> kinetic, genetic are byte-similar in the -netic tail and semantically unrelated. The encoding is doing exactly what its definition says, no more.

dp coverage (fraction of vocab tokens with byte length <= dp):

| Tokenizer | dp = 16 | dp = 32 | dp = 64 |
|---|---|---|---|
| gpt2 | 99.86% | 99.97% | 99.99% |
| SmolLM2-135M | 99.68% | 99.90% | 99.99% |
| Pythia-160M | 99.46% | 99.74% | 99.92% |
| Qwen2.5-0.5B | 98.73% | 99.84% | 99.92% |

dp = 16 covers 98.7% to 99.9% of every vocab tested, which is what justifies using dp = 16 (D = 4096) in the Claim 3 training run. Qwen is lowest at dp = 16 because its 151K vocab contains longer multilingual pieces; dp = 32 recovers it to 99.84%.

OOV strings as single vectors. Technical terms retrieve sensible byte-relatives: asynchronously -> synchron, synchronized, synchronization, Async, sync; deserialization -> specialization, initialization; kubernetes -> Internet, uber, Tube (prefix and interior byte overlap). Fully made-up words do worse: shoggoth -> thought, show, shop (shares the sh prefix only); nibbletron -> inflation, isolation, abolition (the -tion / -ation tail dominates). tiramisu -> thritis, tera, ti. So the capability exists (any UTF-8 string of up to dp bytes gets a vector and lands near byte-similar vocab entries), but its usefulness scales with how much real byte structure the novel string shares with the vocab.

Cross-tokenizer stability. Mean top-5 Jaccard across the six tokenizer pairs, per probe: nation 0.71, magnet 0.59, station 0.58, running 0.56, run 0.44, compute 0.43, computer 0.37, magnetic 0.29. Mean 0.50. Half of any two tokenizers' top-5 neighbourhoods for the same probe are shared, even though the four vocabularies range from 49K to 152K entries and were built with independent merge orders. Agreement is highest where the probe's byte structure is short and common in every vocab (nation) and lowest for longer probes whose pieces get split differently (magnetic).

### Conclusion for Claim 2

Both on-paper properties hold exactly on real vocabularies: unit norm for every one of 50,257 tokens, and cosine (L - k)/L on every same-length pair, to float precision. The encoding produces a well-defined kind of structure, byte-level locality: two strings are close if and only if they share bytes at the same positions. That gives typo robustness (0.86 to 0.88 for a single substitution), case separation (0.0), and a cross-tokenizer-stable neighbourhood structure (Jaccard 0.50). It also gives the predictable failure modes: near-neighbour unrelated words score as high as typos, rhymes cluster with inflections, and insertions are penalized more than substitutions. The rule is now pinned down well enough to build the Claim 3 training comparison on top of it with dp = 16.

### Limitations and why we stopped here

- All checks are on four English-centric tokenizers and English probe words. The dp coverage table hints that non-Latin scripts behave differently (Qwen's drop at dp = 16), but we did not probe any non-Latin strings, emoji, or combining-character sequences. Doing that properly needs a multilingual tokenizer set we could not download within Colab's RAM (the Qwen codec table alone peaked at 3.6 GB).
- The closed-form check is on seven hand-picked pairs. It is exact by construction for same-length pairs, so more pairs would not add information, but we have no equivalent closed form for insertions and deletions and only one measured example (0.73).
- Neighbourhood quality is judged by inspection. There is no strict-family metric in this claim because the codec is being characterized, not scored; the scoring happens in Claims 1 and 5.
- The OOV probe is a capability demonstration on 7 strings, not a benchmark. Whether a trained model can use a single-vector OOV input productively is a training-scale question that Claim 3's model is too small to answer.
- Everything here is pre-training geometry. None of it says whether the encoding helps or hurts a language model. That is Claim 3, and nothing in Claims 1 or 2 can substitute for it.
