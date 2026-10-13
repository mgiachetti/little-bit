"""Full-precision baseline perplexity on the WikiText-2 test set (same windows as train.py's final eval).

    python scripts/teacher_ppl.py facebook/opt-125m Qwen/Qwen3-0.6B
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from littlebit.data import eval_chunks  # noqa: E402
from littlebit.evaluate import perplexity  # noqa: E402
from train import pick_device  # noqa: E402

if __name__ == "__main__":
    device = pick_device("auto")
    for name in sys.argv[1:]:
        tok = AutoTokenizer.from_pretrained(name)
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device)
        test = eval_chunks(tok, min(2048, m.config.max_position_embeddings))
        print(f"| {name} | {len(test)} × {test.shape[1]} | {perplexity(m, test, device):.2f} | "
              f"{perplexity(m, test, device, max_batches=20):.2f} |", flush=True)
