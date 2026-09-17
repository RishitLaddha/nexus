"""
nexus/probe_utils.py
Shared machinery for all nearest-neighbor probes (claims 1, 2, 5):
canonical form, mean-centered cosine retrieval, loose/root/strict morph@K.
"""

import torch


def canonical_form(s: str) -> str:
    # \u2581 = SentencePiece marker, \u0120 = GPT-2 space marker
    s = s.replace("\u2581", " ").replace("\u0120", " ")
    s = s.strip(" \t\n\r.,;:!?\"'`()[]{}_-/\\<>")
    return s.casefold()


@torch.no_grad()
def centered_neighbors(E: torch.Tensor, query_ids, K: int = 10,
                       query_vecs: torch.Tensor = None, exclude_self=True):
    """Mean-centered cosine top-K retrieval over embedding matrix E [V, d].

    query_ids: list of token ids (used for self-exclusion). If query_vecs is
    given, those vectors are used as queries (after centering with E's mean);
    otherwise rows of E are used.
    """
    E = E.float()
    mu = E.mean(0, keepdim=True)
    Ec = torch.nn.functional.normalize(E - mu, dim=-1)
    if query_vecs is None:
        Q = Ec[query_ids]
    else:
        Q = torch.nn.functional.normalize(query_vecs.float() - mu, dim=-1)
    sims = Q @ Ec.T                                   # [nq, V]
    if exclude_self:
        for r, qid in enumerate(query_ids):
            if qid is not None and 0 <= qid < sims.size(1):
                sims[r, qid] = -2.0
    vals, idxs = sims.topk(K, dim=-1)
    return idxs, vals


def loose_morph_at_k(probe: str, retrieved_tokens) -> float:
    """Fraction of retrievals whose canonical form DIFFERS from the probe's
    (escape from typographic clustering). Higher = less typographic."""
    pc = canonical_form(probe)
    esc = [1.0 for t in retrieved_tokens if canonical_form(t) != pc]
    return sum(esc) / max(len(retrieved_tokens), 1)


def root_morph_at_k(root: str, retrieved_tokens) -> float:
    r = root.casefold()
    hit = [1.0 for t in retrieved_tokens if r in canonical_form(t)]
    return sum(hit) / max(len(retrieved_tokens), 1)


def strict_morph_at_k(family: set, retrieved_tokens) -> float:
    fam = {f.casefold() for f in family}
    hit = [1.0 for t in retrieved_tokens if canonical_form(t) in fam]
    return sum(hit) / max(len(retrieved_tokens), 1)


def anisotropy(E: torch.Tensor, n_pairs: int = 20000, seed: int = 0):
    """||mu|| of the embedding table and raw mean pairwise cosine (uncentered),
    estimated on random pairs. Mirrors the paper's Table 6 diagnostic."""
    g = torch.Generator().manual_seed(seed)
    E = E.float()
    mu_norm = E.mean(0).norm().item()
    V = E.size(0)
    i = torch.randint(0, V, (n_pairs,), generator=g)
    j = torch.randint(0, V, (n_pairs,), generator=g)
    a = torch.nn.functional.normalize(E[i], dim=-1)
    b = torch.nn.functional.normalize(E[j], dim=-1)
    return mu_norm, (a * b).sum(-1).mean().item()


# Probe families used by claims 1 and 2 (same four as the paper)
PROBE_FAMILIES = {
    "run":     ["run", "runs", "running", "runner", "ran"],
    "compute": ["compute", "computer", "computing", "computation", "computes"],
    "magnet":  ["magnet", "magnets", "magnetic", "magnetize", "magnetized"],
    "tion":    ["nation", "station", "action", "rotation", "creation"],
}

# Hand-curated strict families for the layered probe (claim 5)
STRICT_FAMILIES = {
    "run": {
        "root": "run",
        "probes": ["run", "runs", "running", "runner"],
        "family": ["run", "runs", "running", "runner", "runners", "ran",
                   "rerun", "reruns", "runnable", "outrun", "overrun"],
    },
    "compute": {
        "root": "comput",
        "probes": ["compute", "computer", "computing", "computed"],
        "family": ["compute", "computes", "computed", "computing", "computer",
                   "computers", "computation", "computations",
                   "computational", "recompute"],
    },
    "nation": {
        "root": "nation",
        "probes": ["nation", "national", "nations", "nationally"],
        "family": ["nation", "nations", "national", "nationals", "nationally",
                   "nationality", "nationalities", "nationalism",
                   "nationalist", "international", "multinational"],
    },
    "magnet": {
        "root": "magnet",
        "probes": ["magnet", "magnetic", "magnets", "magnetism"],
        "family": ["magnet", "magnets", "magnetic", "magnetism", "magnetize",
                   "magnetized", "magnetization", "electromagnet",
                   "electromagnetic", "magnetically"],
    },
}
