"""
CLAIM 3 analysis - turns runs/*/log.json into the paper-style artifacts:

  results/claim3/val_loss_curves.png     (both arms, all seeds)
  results/claim3/gap_curve.png           (BPE - NibbleNet gap vs step)
  results/claim3/trajectory_table.csv    (paper Table 7 analog)
  results/claim3/final_summary.csv       (paper Table 8 analog: mean/std/SNR)
  results/claim3/sample_efficiency.txt   (steps to reach BPE final loss)
  results/claim3/mechanism_table.csv     (paper Table 9 analog: emb std walk)
  results/claim3/param_accounting.csv    (paper Sec 6.9 analog, both framings)

Run after at least one seed of each arm has finished:
    python claim3_analyze.py
"""

import csv, glob, json, os
import numpy as np


def load_runs(pattern="runs/*/log.json"):
    runs = {"bpe": {}, "nibble": {}}
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            log = json.load(f)
        arm = log["config"]["arm"]
        seed = log["config"]["seed"]
        runs[arm][seed] = log
    return runs


def main(outdir="results/claim3"):
    os.makedirs(outdir, exist_ok=True)
    runs = load_runs()
    seeds = sorted(set(runs["bpe"]) & set(runs["nibble"]))
    assert seeds, "need at least one finished seed of BOTH arms in runs/"
    print(f"paired seeds: {seeds}")

    ref = runs["bpe"][seeds[0]]["steps"]

    def series(arm):
        return np.array([runs[arm][s]["val_loss"] for s in seeds])  # [S, T]

    bpe, nibble = series("bpe"), series("nibble")
    gap = bpe - nibble                                   # >0 means NibbleNet better

    # ---- Table 7 analog: trajectory at round checkpoints ----
    marks = [i for i, st in enumerate(ref)
             if st in (200, 500, 1000, 1500, 2000, 2500) or i == len(ref) - 1]
    with open(f"{outdir}/trajectory_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "nibble_val", "bpe_val", "gap_nats", "nibble_lower_%"])
        for i in marks:
            kb, bb = nibble[:, i].mean(), bpe[:, i].mean()
            w.writerow([ref[i], round(kb, 4), round(bb, 4),
                        round(bb - kb, 4), round(100 * (bb - kb) / bb, 2)])
            print(f"step {ref[i]:5d}  nibble {kb:.4f}  bpe {bb:.4f}  "
                  f"gap {bb-kb:+.4f}")

    # ---- Table 8 analog: final summary across seeds ----
    fk, fb = nibble[:, -1], bpe[:, -1]
    g = fb - fk
    snr = abs(g.mean()) / (g.std() + 1e-12) if len(seeds) > 1 else float("nan")
    fav = int((gap > 0).sum())
    with open(f"{outdir}/final_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stat", "nibble", "bpe", "gap", "nibble_lower_%"])
        w.writerow(["mean", round(fk.mean(), 4), round(fb.mean(), 4),
                    round(g.mean(), 4), round(100 * g.mean() / fb.mean(), 2)])
        w.writerow(["std", round(fk.std(), 4), round(fb.std(), 4),
                    round(g.std(), 4), ""])
        w.writerow(["SNR", "", "", "" if np.isnan(snr) else round(snr, 1), ""])
        w.writerow(["favorable_cells", f"{fav}/{gap.size}", "", "", ""])
    print(f"\nfinal: nibble {fk.mean():.4f}  bpe {fb.mean():.4f}  "
          f"gap {g.mean():+.4f} ({100*g.mean()/fb.mean():+.2f}%)  "
          f"favorable {fav}/{gap.size}")

    # ---- sample efficiency: steps for nibble to reach bpe's final loss ----
    target = fb.mean()
    km = nibble.mean(0)
    idx = np.argmax(km <= target) if (km <= target).any() else None
    with open(f"{outdir}/sample_efficiency.txt", "w") as f:
        if idx:
            ratio = ref[-1] / max(ref[idx], 1)
            msg = (f"BPE final val loss {target:.4f} reached by NibbleNet at step "
                   f"{ref[idx]} vs BPE's {ref[-1]} -> {ratio:.2f}x fewer steps")
        else:
            msg = (f"NibbleNet did not reach BPE's final loss {target:.4f} "
                   f"within the schedule (report honestly).")
        f.write(msg + "\n")
        print(msg)

    # ---- Table 9 analog: mechanism (embedding scale walk) ----
    with open(f"{outdir}/mechanism_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "bpe_emb_std", "nibble_proj_scale"])
        eb = np.array([runs["bpe"][s]["emb_scale"] for s in seeds]).mean(0)
        ek = np.array([runs["nibble"][s]["emb_scale"] for s in seeds]).mean(0)
        for i in range(0, len(ref), max(1, len(ref) // 8)):
            w.writerow([ref[i], round(eb[i], 4), round(ek[i], 4)])
    print(f"mechanism: bpe emb std {eb[0]:.4f} -> {eb[-1]:.4f}   "
          f"nibble proj scale {ek[0]:.4f} -> {ek[-1]:.4f}")

    # ---- parameter accounting, both framings (honest version) ----
    pb = runs["bpe"][seeds[0]]["params"]
    pk = runs["nibble"][seeds[0]]["params"]
    with open(f"{outdir}/param_accounting.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["framing", "bpe_tied", "nibble", "note"])
        w.writerow(["input-side trainable",
                    pb["input_side_trainable"], pk["input_side_trainable"],
                    f"{100*(1-pk['input_side_trainable']/pb['input_side_trainable']):.0f}% cut"])
        w.writerow(["TOTAL trainable",
                    pb["total_trainable"], pk["total_trainable"],
                    "nibble pays an untied lm_head at this scale"])

    # ---- plots ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4.5))
    for s in seeds:
        plt.plot(ref, runs["bpe"][s]["val_loss"], "C0", alpha=0.7,
                 label="BPE-tied" if s == seeds[0] else None)
        plt.plot(ref, runs["nibble"][s]["val_loss"], "C1", alpha=0.7,
                 label="NibbleNet" if s == seeds[0] else None)
    plt.xlabel("step"); plt.ylabel("val loss"); plt.legend()
    plt.title("Controlled comparison: identical model/data/schedule")
    plt.tight_layout(); plt.savefig(f"{outdir}/val_loss_curves.png", dpi=150)

    plt.figure(figsize=(8, 3.5))
    plt.plot(ref, gap.mean(0), "k")
    if len(seeds) > 1:
        plt.fill_between(ref, gap.mean(0) - gap.std(0),
                         gap.mean(0) + gap.std(0), alpha=0.2)
    plt.axhline(0, color="grey", lw=0.8)
    plt.xlabel("step"); plt.ylabel("gap (BPE - NibbleNet), nats")
    plt.title("Validation-loss gap through training")
    plt.tight_layout(); plt.savefig(f"{outdir}/gap_curve.png", dpi=150)
    print(f"\nDone. Artifacts in {outdir}/")


if __name__ == "__main__":
    main()
