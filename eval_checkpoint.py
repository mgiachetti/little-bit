"""Load a checkpoint saved by train.py, report WikiText-2 perplexity and sample generations.

    python eval_checkpoint.py runs/Qwen3-0.6B-bpw0.55-itq-msenorm-skip1-c4/littlebit.pt
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from littlebit import bpw_report, convert_model
from littlebit.data import eval_chunks
from littlebit.evaluate import generate, perplexity
from train import pick_device

PROMPTS = ["Leonardo da Vinci painted the Mona Lisa. The Mona Lisa is",
           "Turing is called the father of computer science. Computer science is"]


def load(path: str, device) -> tuple:
    ck = torch.load(path, map_location="cpu")
    a = ck["args"]
    tok = AutoTokenizer.from_pretrained(a["model"])
    model = AutoModelForCausalLM.from_pretrained(a["model"], dtype=torch.float32)
    # Rebuild the same LittleBit structure (same ranks), then load the trained values.
    convert_model(model, a["bpw"], residual=not a["no_residual"], kv_rank_mult=a["kv_rank_mult"],
                  init="empty", verbose=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), tok, a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--eval-len", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    model, tok, train_args = load(args.checkpoint, device)
    test = eval_chunks(tok, min(args.eval_len, model.config.max_position_embeddings))
    print(json.dumps({"model": train_args["model"], "bpw": train_args["bpw"], "init": train_args.get("init", "svid"),
                      **bpw_report(model)}))
    print(f"WikiText-2 test PPL ({len(test)} windows × {test.shape[1]}): {perplexity(model, test, device):.2f}")
    for prompt in PROMPTS:
        print("\n" + generate(model, tok, prompt, device, args.max_new_tokens))


if __name__ == "__main__":
    main()
