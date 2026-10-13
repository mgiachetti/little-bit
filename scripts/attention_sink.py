"""Profile hidden-state magnitudes per layer to find massive activations ("attention sinks").

    python scripts/attention_sink.py --model Qwen/Qwen3-0.6B > results/qwen3_attention_sink.md
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from littlebit.data import eval_chunks  # noqa: E402
from train import pick_device  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--seqs", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    ids = eval_chunks(tok, args.seq_len)[:args.seqs].to(device)
    with torch.no_grad():
        hs = m.base_model(ids, output_hidden_states=True).hidden_states

    print(f"{args.model}: hidden states on {args.seqs} WikiText-2 windows × {args.seq_len} tokens "
          f"(index 0 = embeddings, i = output of block i-1)\n")
    print("| idx | max abs | RMS | top-3 dims | energy in top-3 dims | energy at token 0 |")
    print("|---|---|---|---|---|---|")
    for i, h in enumerate(hs):
        per_dim = h.abs().amax((0, 1))
        top = per_dim.topk(3).indices.tolist()
        e = h.pow(2)
        share_dims = e.sum((0, 1))[top].sum() / e.sum()
        share_tok0 = e[:, 0].sum() / e.sum()
        print(f"| {i} | {h.abs().max().item():.1f} | {h.pow(2).mean().sqrt().item():.2f} | {top} | "
              f"{100 * share_dims.item():.1f}% | {100 * share_tok0.item():.1f}% |")

    # Which sub-module of block 2 writes the spike?
    contrib = {}
    for name in ("self_attn", "mlp"):
        def hook(mod, inp, out, name=name):
            o = out[0] if isinstance(out, tuple) else out
            contrib[name] = o[0, 0].abs().max().item()
        getattr(m.base_model.layers[2], name).register_forward_hook(hook)
    with torch.no_grad():
        m.base_model(ids[:1])
    print(f"\nBlock 2 contribution (max |x| at token 0): {contrib}")


if __name__ == "__main__":
    main()
