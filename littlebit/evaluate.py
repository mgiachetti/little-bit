"""Perplexity on WikiText-2 (non-overlapping windows, as in GPTQ/OmniQuant/STBLLM)."""

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def perplexity(model, chunks: torch.Tensor, device, batch_size: int = 1, max_batches: int | None = None,
               token_chunk: int = 512) -> float:
    """Logits are computed token_chunk positions at a time to bound memory for large vocabularies."""
    model.eval()
    head = model.get_output_embeddings()
    nll, count = 0.0, 0
    for i, start in enumerate(range(0, len(chunks), batch_size)):
        if max_batches is not None and i >= max_batches:
            break
        ids = chunks[start:start + batch_size].to(device)
        hidden = model.base_model(ids).last_hidden_state[:, :-1].flatten(0, 1)
        targets = ids[:, 1:].flatten()
        for j in range(0, hidden.size(0), token_chunk):
            logits = head(hidden[j:j + token_chunk]).float()
            nll += F.cross_entropy(logits, targets[j:j + token_chunk], reduction="sum").item()
        count += targets.numel()
    return math.exp(nll / count)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device, max_new_tokens: int = 60) -> str:
    model.eval()
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)
