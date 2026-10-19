"""Speed of one LittleBit MoE layer shaped like Qwen3-30B-A3B: per-expert loop vs. grouped GEMM.

Random weights and routing (no download). Times forward + backward of the experts only.

    python scripts/bench_moe.py --tokens 4096 --precision bf16
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from littlebit import LittleBitExperts, LittleBitLinear, rank_for_bpw


class FakeExperts(nn.Module):
    def __init__(self, e, h, i):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(e, 2 * i, h) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(e, h, i) * 0.02)
        self.act_fn = F.silu


class LoopExperts(nn.Module):
    """One LittleBitLinear per expert matrix, run in a Python loop like transformers' eager experts."""

    def __init__(self, experts, r_gu, r_dn, init):
        super().__init__()
        def lin(w, r):
            m = nn.Linear(w.shape[1], w.shape[0], bias=False)
            m.weight.data = w.detach()
            return LittleBitLinear(m, r, init=init)
        self.gate_up = nn.ModuleList(lin(w, r_gu) for w in experts.gate_up_proj)
        self.down = nn.ModuleList(lin(w, r_dn) for w in experts.down_proj)
        self.act_fn = experts.act_fn

    def forward(self, x, idx, w):
        out = torch.zeros_like(x)
        mask = F.one_hot(idx, len(self.gate_up)).permute(2, 1, 0)
        for e in mask.sum((1, 2)).nonzero()[:, 0].tolist():
            slot, tok = torch.where(mask[e])
            gate, up = self.gate_up[e](x[tok]).chunk(2, -1)
            y = self.down[e](self.act_fn(gate) * up) * w[tok, slot, None]
            out.index_add_(0, tok, y.to(out.dtype))
        return out


def bench(fn, iters):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=768)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--bpw", type=float, default=0.55)
    p.add_argument("--init", default="itq")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--skip-loop", action="store_true")
    args = p.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    e, h, i = args.experts, args.hidden, args.inter
    r_gu = rank_for_bpw(args.bpw, 2 * i, h)
    r_dn = rank_for_bpw(args.bpw, h, i)
    experts = FakeExperts(e, h, i).to(dev)
    x = torch.randn(args.tokens, h, device=dev)
    router = torch.randn(h, e, device=dev) / h ** 0.5
    w, idx = torch.topk(torch.softmax(x @ router, -1), args.top_k, -1)
    w = w / w.sum(-1, keepdim=True)
    print(f"{e} experts, H={h}, I={i}, top-{args.top_k}, {args.tokens} tokens, ranks {r_gu}/{r_dn}, "
          f"{args.precision}, {torch.cuda.get_device_name()}")

    builders = {"grouped": lambda: LittleBitExperts(experts, r_gu, r_dn, init=args.init)}
    if not args.skip_loop:
        builders["loop"] = lambda: LoopExperts(experts, r_gu, r_dn, args.init)
    results = {}
    for name, build in builders.items():  # one implementation on the GPU at a time
        t = time.perf_counter()
        mod = build()
        torch.cuda.synchronize()
        init_s = time.perf_counter() - t

        def step():
            with torch.autocast("cuda", torch.bfloat16, enabled=args.precision == "bf16"):
                out = mod(x, idx, w)
            out.float().pow(2).mean().backward()
            mod.zero_grad(set_to_none=True)
        experts.cpu()  # dense weights are only needed for the init
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        results[name] = bench(step, args.iters)
        print(f"{name:8s} init {init_s:6.1f}s   fwd+bwd {results[name] * 1e3:8.1f} ms   "
              f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
        del mod
        experts.to(dev)
        torch.cuda.empty_cache()
    if "loop" in results:
        print(f"speed-up: {results['loop'] / results['grouped']:.1f}x")


if __name__ == "__main__":
    main()
