"""Reconstruction error of the LittleBit initialization, per matrix type, straight from safetensors.

Downloads only the shards that hold the requested layers (no model instantiation), so it also
works for MoE checkpoints far larger than GPU memory. Reports, per matrix type at the target BPW:
  init  ||Ŵ - W|| / ||W||   for the binarized primary + residual init (what training starts from)
  svd   ||W_r - W|| / ||W||  for the best unquantized rank-r approximation (r = the latent rank)
MoE experts are stacked like LittleBitExperts does it (gate and up fused, one row per expert).

    python scripts/recon_error.py Qwen/Qwen3-30B-A3B --layers 0 23 46
"""

import argparse
import json
from collections import defaultdict

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from littlebit import rank_for_bpw
from littlebit.experts import LittleBitGroupedLinear


def rel_err(approx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Relative Frobenius error per leading (expert) index."""
    return (approx - w).flatten(1).norm(dim=1) / w.flatten(1).norm(dim=1)


@torch.no_grad()
def measure(w: torch.Tensor, bpw: float, init: str, chunk: int = 16) -> dict:
    """w [E, d_out, d_in] (E = 1 for a dense matrix)."""
    _, d_out, d_in = w.shape
    r = min(rank_for_bpw(bpw, d_out, d_in), d_out, d_in)
    init_err, svd_err = [], []
    for i in range(0, w.shape[0], chunk):
        wc = w[i:i + chunk].cuda().float()
        lb = LittleBitGroupedLinear(wc, r, init=init)
        init_err.append(rel_err(lb.effective_weight(), wc))
        u, s, vh = torch.linalg.svd(wc, full_matrices=False)
        svd_err.append(rel_err((u[..., :r] * s[..., None, :r]) @ vh[..., :r, :], wc))
    init_err, svd_err = torch.cat(init_err), torch.cat(svd_err)
    return {"shape": f"{d_out}x{d_in}", "experts": w.shape[0], "rank": r,
            "init": init_err.mean().item(), "init_max": init_err.max().item(), "svd": svd_err.mean().item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("--layers", type=int, nargs="+", default=[0])
    p.add_argument("--bpw", type=float, default=0.55)
    p.add_argument("--init", default="itq")
    p.add_argument("--out", default=None, help="append JSON lines here")
    args = p.parse_args()

    try:
        index = json.load(open(hf_hub_download(args.repo, "model.safetensors.index.json")))["weight_map"]
    except Exception:  # single-file checkpoint
        f = hf_hub_download(args.repo, "model.safetensors")
        with safe_open(f, "pt") as sf:
            index = {k: "model.safetensors" for k in sf.keys()}
    files = {}

    def get(key: str) -> torch.Tensor:
        fn = index[key]
        if fn not in files:
            files[fn] = safe_open(hf_hub_download(args.repo, fn), "pt")
        return files[fn].get_tensor(key)

    for layer in args.layers:
        pre = f"model.layers.{layer}."
        keys = [k for k in index if k.startswith(pre) and k.endswith(".weight")]
        mats: dict[str, torch.Tensor] = {}
        experts = defaultdict(dict)
        for k in keys:
            name = k[len(pre):-len(".weight")]
            if ".experts." in name:  # mlp.experts.{e}.gate_proj
                e, proj = name.split(".experts.")[1].split(".")
                experts[proj][int(e)] = k
            elif name.endswith("_proj") and "gate." not in name:
                mats[name] = get(k)[None]
        if experts:
            n = len(experts["gate_proj"])
            gu = [torch.cat([get(experts["gate_proj"][e]), get(experts["up_proj"][e])]) for e in range(n)]
            mats["mlp.experts.gate_up"] = torch.stack(gu)
            mats["mlp.experts.down"] = torch.stack([get(experts["down_proj"][e]) for e in range(n)])
        for name, w in sorted(mats.items()):
            torch.manual_seed(0)
            res = {"repo": args.repo, "layer": layer, "matrix": name, "bpw": args.bpw, **measure(w, args.bpw, args.init)}
            print(f"{args.repo:22s} L{layer:<3d} {name:22s} {res['shape']:>10s} ×{res['experts']:<4d} r={res['rank']:<4d} "
                  f"init {res['init']:.3f} (max {res['init_max']:.3f})  svd-r {res['svd']:.3f}", flush=True)
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
