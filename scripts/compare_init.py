"""Reconstruction error ||W - Ŵ|| / ||W|| of each initialization (no training).

    python scripts/compare_init.py --model facebook/opt-125m > results/init_reconstruction.md
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from littlebit import LittleBitLinear, rank_for_bpw  # noqa: E402

INITS = ("svid", "rot", "itq")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--blocks", default="0,6,11", help="transformer blocks to sample")
    p.add_argument("--bpw", default="1.0,0.55,0.3,0.1")
    args = p.parse_args()

    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    blocks = [f".layers.{b}." for b in args.blocks.split(",")]
    layers = [(n, mod) for n, mod in m.named_modules()
              if isinstance(mod, torch.nn.Linear) and any(b in n for b in blocks)]
    print(f"{args.model}: mean relative error over {len(layers)} linear layers (blocks {args.blocks})\n")
    print("| BPW | " + " | ".join(INITS) + " |")
    print("|---|" + "---|" * len(INITS))
    for bpw in map(float, args.bpw.split(",")):
        row = []
        for init in INITS:
            errs = []
            for _, lin in layers:
                w = lin.weight.detach()
                r = rank_for_bpw(bpw, lin.out_features, lin.in_features)
                errs.append(((w - LittleBitLinear(lin, r, init=init).effective_weight()).norm() / w.norm()).item())
            row.append(sum(errs) / len(errs))
        print(f"| {bpw} | " + " | ".join(f"{e:.4f}" for e in row) + " |", flush=True)


if __name__ == "__main__":
    main()
