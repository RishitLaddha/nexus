"""
CLAIM 3 - controlled training comparison (the central bet).

Trains the SAME scaled-down GPT under identical data/schedule/architecture,
with the input embedding pathway as the ONLY difference:

    python claim3_train.py --arm bpe  --seed 1337
    python claim3_train.py --arm nibble --seed 1337

Model: 6 layers, 6 heads, d=384, ctx 256, GPT-2 vocab 50257.
  Why this shape: it is nanoGPT's GPT-2 architecture scaled down, and the
  tied embedding table (50257 x 384 = 19.3M) still DOMINATES the 10.6M body,
  i.e. we stay in the input-heavy regime where the paper's question is live.
Budget: 3000 steps x 16,384 tokens/step ~= 49M tokens, val every 100 steps
  on a held-out shard, exactly the paper's evaluation style (val loss, not
  train loss). ~2-2.5 h/run on a Colab T4, ~45-60 min on an EC2 g5.xlarge.

Also logs the mechanism observation every eval (paper Table 9 analog):
  bpe arm  -> std of the embedding table (expect upward walk from 0.02)
  nibble arm -> std of the projection * sqrt(D) (expect stable ~1.0)

Outputs: runs/{arm}_s{seed}/log.json + ckpt.pt
"""

import argparse, json, math, os, time
import numpy as np
import torch

from nexus.model import GPT, GPTConfig
from nexus.codec import build_byte_buffer


def get_batch(data, block, bs, device, gen):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(
        data[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(
        data[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def eval_loss(model, data, block, bs, device, iters=40):
    model.eval()
    gen = torch.Generator().manual_seed(123)  # SAME val batches for both arms
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, block, bs, device, gen)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(device == "cuda")):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def emb_scale(model):
    if model.cfg.embedding == "bpe":
        return model.wte.weight.detach().float().std().item()
    # projection std in units where init == 1.0 (init std = 1/sqrt(D))
    w = model.wte.proj.weight.detach().float()
    return (w.std() * math.sqrt(model.wte.D)).item()


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train = np.memmap(f"{args.data}/train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(f"{args.data}/val.bin", dtype=np.uint16, mode="r")

    cfg = GPTConfig(vocab_size=50257, block_size=args.block,
                    n_layer=args.n_layer, n_head=args.n_head,
                    n_embd=args.n_embd, embedding=args.arm, dp=args.dp)
    if args.arm == "nibble":
        from transformers import AutoTokenizer
        tb, tl = build_byte_buffer(AutoTokenizer.from_pretrained("gpt2"),
                                   args.dp)
        model = GPT(cfg, tb, tl).to(device)
    else:
        model = GPT(cfg).to(device)
    print(json.dumps(model.param_report(), indent=2, default=str))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    gen = torch.Generator().manual_seed(args.seed)

    def lr_at(it):
        if it < args.warmup:
            return args.lr * (it + 1) / args.warmup
        r = (it - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.1 + 0.5 * (args.lr * 0.9) * (1 + math.cos(math.pi * r))

    tag = f"{args.arm}_s{args.seed}"
    outdir = f"{args.out}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    log = {"config": vars(args), "params": model.param_report(),
           "steps": [], "val_loss": [], "emb_scale": [], "step_time_ms": []}

    model.train()
    t_run = time.time()
    for it in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = get_batch(train, args.block, args.micro_bs, device, gen)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(device == "cuda")):
                _, loss = model(x, y)
            scaler.scale(loss / args.grad_accum).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        dt = (time.time() - t0) * 1000

        if it % args.eval_every == 0 or it == args.steps - 1:
            vl = eval_loss(model, val, args.block, args.micro_bs, device)
            es = emb_scale(model)
            log["steps"].append(it)
            log["val_loss"].append(vl)
            log["emb_scale"].append(es)
            log["step_time_ms"].append(dt)
            el = (time.time() - t_run) / 60
            print(f"[{tag}] step {it:5d}  val_loss {vl:.4f}  "
                  f"emb_scale {es:.4f}  {dt:.0f} ms/step  {el:.1f} min")
            with open(f"{outdir}/log.json", "w") as f:
                json.dump(log, f, indent=2)

    torch.save({"model": model.state_dict(), "config": vars(args)},
               f"{outdir}/ckpt.pt")
    print(f"saved {outdir}/ckpt.pt and log.json  "
          f"(total {(time.time()-t_run)/60:.1f} min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["bpe", "nibble"], required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--micro_bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=4)  # 16*256*4 = 16384 tok/step
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=384)
    ap.add_argument("--dp", type=int, default=16)
    main(ap.parse_args())
