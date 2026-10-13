"""LittleBit QAT with knowledge distillation (Section 4.1, Eq. 11).

L = KL(teacher || student) on logits + λ · MSE on hidden states of every block.

    python train.py --model facebook/opt-125m --bpw 0.55 --steps 2000
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from littlebit import bpw_report, convert_model
from littlebit.data import eval_chunks, train_chunks
from littlebit.evaluate import generate, perplexity


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _chunk_kl(s_h, t_h, s_head, t_head):
    s_logp = F.log_softmax(s_head(s_h).float(), -1)
    with torch.no_grad():
        t_logp = F.log_softmax(t_head(t_h).float(), -1)
    return F.kl_div(s_logp, t_logp, log_target=True, reduction="sum")


def kd_loss(student, teacher, ids, lam: float, mse_norm: bool = False, mse_skip: int = 0,
            chunk: int = 128):
    """KL on logits + λ·MSE on hidden states.

    Logits are computed per chunk of tokens under checkpointing, so the full
    (tokens × vocab) tensors never coexist in memory — needed for 150k vocabularies.
    mse_norm divides each layer's MSE by the teacher's mean squared activation
    (not in the paper; keeps models with massive activations from drowning the KL).
    mse_skip drops the first positions from the MSE: models like Qwen3 park a ~7000-valued
    "attention sink" activation on token 0 that otherwise holds >99% of the hidden-state energy.
    """
    with torch.no_grad():
        t_out = teacher.base_model(ids, output_hidden_states=True)
    s_out = student.base_model(ids, output_hidden_states=True)
    s_last = s_out.last_hidden_state.flatten(0, 1)
    t_last = t_out.last_hidden_state.flatten(0, 1)
    s_head, t_head = student.get_output_embeddings(), teacher.get_output_embeddings()
    kl = sum(checkpoint(_chunk_kl, s_last[i:i + chunk], t_last[i:i + chunk], s_head, t_head,
                        use_reentrant=False)
             for i in range(0, s_last.size(0), chunk)) / s_last.size(0)
    # Skip index 0 (embedding output, identical for both models).
    pairs = [(s[:, mse_skip:], t[:, mse_skip:])
             for s, t in zip(s_out.hidden_states[1:], t_out.hidden_states[1:])]
    mse = sum(F.mse_loss(s.float(), t.float()) / (t.float().pow(2).mean() if mse_norm else 1)
              for s, t in pairs) / len(pairs)
    return kl + lam * mse, kl, mse


def lr_at(step: int, total: int, peak: float, warmup_frac: float) -> float:
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


def share_embeddings(student, teacher):
    """Point the student's (frozen, full-precision) embeddings and head at the teacher's tensors."""
    student.get_input_embeddings().weight = teacher.get_input_embeddings().weight
    student.get_output_embeddings().weight = teacher.get_output_embeddings().weight


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--bpw", type=float, default=0.55)
    p.add_argument("--no-residual", action="store_true")
    p.add_argument("--kv-rank-mult", type=float, default=4.0)
    p.add_argument("--init", choices=["svid", "rot", "itq"], default="svid",
                   help="svid = LittleBit, itq = LittleBit-2 Joint-ITQ")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1.2e-4)
    p.add_argument("--warmup", type=float, default=0.02)
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--mse-norm", action="store_true", help="relative per-layer MSE (not in the paper)")
    p.add_argument("--mse-skip-first", type=int, default=0,
                   help="exclude the first N positions (attention sink) from the hidden-state MSE")
    p.add_argument("--c4-mtokens", type=float, default=0, help="mix this many million C4 tokens into training")
    p.add_argument("--wiki-repeat", type=int, default=1, help="repeat WikiText-2 train this many times in the mix")
    p.add_argument("--train-embeddings", action="store_true",
                   help="also train embeddings/lm_head (default: frozen and shared with the teacher)")
    p.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing in the student")
    p.add_argument("--eval-len", type=int, default=2048)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=40, help="cap for periodic evals (final eval is full)")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="runs")
    p.add_argument("--tag", default="", help="suffix for the run directory")
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="save a resumable checkpoint every N steps; an existing one is resumed automatically")
    args = p.parse_args()

    torch.manual_seed(0)
    device = pick_device(args.device)
    run_dir = Path(args.out) / (f"{args.model.split('/')[-1]}-bpw{args.bpw}-{args.init}"
                               + ("-msenorm" if args.mse_norm else "")
                               + (f"-skip{args.mse_skip_first}" if args.mse_skip_first else "")
                               + (f"-{args.tag}" if args.tag else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt.pt"
    tok = AutoTokenizer.from_pretrained(args.model)
    teacher = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    teacher.requires_grad_(False)
    student = copy.deepcopy(teacher)
    teacher.to(device)

    eval_len = min(args.eval_len, teacher.config.max_position_embeddings)
    test = eval_chunks(tok, eval_len)
    print(f"device={device}  teacher PPL (first {args.eval_batches} windows): "
          f"{perplexity(teacher, test, device, max_batches=args.eval_batches):.2f}")

    t0 = time.time()
    student.to(device)
    convert_model(student, args.bpw, residual=not args.no_residual, kv_rank_mult=args.kv_rank_mult,
                  init=args.init)
    student.requires_grad_(True)
    if not args.train_embeddings:
        share_embeddings(student, teacher)
    if args.grad_ckpt:
        student.config.use_cache = False
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rep = bpw_report(student)
    print(f"converted in {time.time() - t0:.1f}s  {json.dumps(rep)}")
    print(f"init PPL (no training): "
          f"{perplexity(student, test, device, max_batches=args.eval_batches):.2f}")

    data = train_chunks(tok, args.seq_len, int(args.c4_mtokens * 1e6), args.wiki_repeat)
    tokens_per_step = args.seq_len * args.batch_size * args.grad_accum
    print(f"train sequences: {len(data)} × {args.seq_len} tokens  "
          f"({args.steps * tokens_per_step / len(data) / args.seq_len:.2f} epochs over {args.steps} steps)")
    params = [p for p in student.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M")
    opt = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))

    start, cursor = 0, 0
    if args.ckpt_every and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        student.load_state_dict(ck["trainable"], strict=False)
        opt.load_state_dict(ck["opt"])
        start, cursor = ck["step"], ck["cursor"]
        print(f"resumed from {ckpt_path} at step {start}", flush=True)
    names = {id(p): n for n, p in student.named_parameters()}
    t0 = time.time()
    for step in range(start, args.steps):
        student.train()
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        for _ in range(args.grad_accum):
            if cursor + args.batch_size > len(data):
                cursor = 0
            ids = data[cursor:cursor + args.batch_size].to(device)
            cursor += args.batch_size
            loss, kl, mse = kd_loss(student, teacher, ids, args.lam, args.mse_norm, args.mse_skip_first)
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 20 == 0:
            mem = f"  mem {torch.cuda.max_memory_allocated() / 2**30:.2f}G" if device.type == "cuda" else ""
            print(f"step {step:5d}  loss {loss.item():.4f}  kl {kl.item():.4f}  mse {mse.item():.5f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  {time.time() - t0:.0f}s{mem}", flush=True)
        if (step + 1) % args.eval_every == 0 and step + 1 < args.steps:
            ppl = perplexity(student, test, device, max_batches=args.eval_batches)
            print(f"step {step + 1}  eval PPL {ppl:.2f}", flush=True)
        if args.ckpt_every and (step + 1) % args.ckpt_every == 0:
            state = {names[id(p)]: p.detach() for p in params}
            torch.save({"trainable": state, "opt": opt.state_dict(), "step": step + 1, "cursor": cursor},
                       ckpt_path.with_suffix(".tmp"))
            ckpt_path.with_suffix(".tmp").replace(ckpt_path)  # atomic: a crash never leaves a torn file

    ppl = perplexity(student, test, device)
    print(f"final WikiText-2 PPL: {ppl:.2f}  ({json.dumps(rep)})")
    student.config.use_cache = True
    for prompt in ["Leonardo da Vinci painted the Mona Lisa. The Mona Lisa is",
                   "Turing is called the father of computer science. Computer science is"]:
        print("\n" + generate(student, tok, prompt, device))

    torch.save({"args": vars(args), "state_dict": student.state_dict(), "ppl": ppl, "report": rep},
               run_dir / "littlebit.pt")
    print(f"\nsaved {run_dir / 'littlebit.pt'}")


if __name__ == "__main__":
    main()
