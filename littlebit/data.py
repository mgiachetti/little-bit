"""WikiText-2 / C4 token streams chunked into fixed-length sequences."""

from pathlib import Path

import torch
from datasets import load_dataset

CACHE = Path.home() / ".cache" / "littlebit"
C4_SHARD = "en/c4-train.{:05d}-of-01024.json.gz"


def _wikitext(split: str) -> str:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(ds["text"])


def _c4_tokens(tokenizer, n_tokens: int, batch: int = 1000) -> torch.Tensor:
    """First n_tokens of C4 (en), documents separated by a blank line. Cached per tokenizer."""
    slug = tokenizer.name_or_path.replace("/", "--")
    path = CACHE / f"{slug}-c4-{n_tokens}.pt"
    if path.exists():
        return torch.load(path)
    sep = tokenizer("\n\n", add_special_tokens=False).input_ids
    parts, total, shard = [], 0, 0
    while total < n_tokens:
        ds = load_dataset("allenai/c4", data_files=C4_SHARD.format(shard), split="train")
        for i in range(0, len(ds), batch):
            for ids in tokenizer(ds[i:i + batch]["text"], add_special_tokens=False).input_ids:
                parts.append(torch.tensor(ids + sep, dtype=torch.int32))
                total += len(ids) + len(sep)
            print(f"\r[c4] shard {shard}: {total / 1e6:.1f}M / {n_tokens / 1e6:.0f}M tokens", end="", flush=True)
            if total >= n_tokens:
                break
        shard += 1
    print()
    ids = torch.cat(parts)[:n_tokens]
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(ids, path)
    return ids


def token_chunks(tokenizer, text: str, seq_len: int) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n = ids.numel() // seq_len
    return ids[: n * seq_len].view(n, seq_len)


def _chunk(ids: torch.Tensor, seq_len: int) -> torch.Tensor:
    n = ids.numel() // seq_len
    return ids[: n * seq_len].view(n, seq_len)


def train_chunks(tokenizer, seq_len: int, c4_tokens: int = 0, wiki_repeat: int = 1) -> torch.Tensor:
    """Training sequences: WikiText-2 train (repeated wiki_repeat times) mixed with C4
    (paper, Section 4.1), shuffled with a fixed seed so resumed runs see the same order."""
    chunks = [token_chunks(tokenizer, _wikitext("train"), seq_len)] * wiki_repeat
    if c4_tokens:
        chunks.append(_chunk(_c4_tokens(tokenizer, c4_tokens), seq_len).long())
    chunks = torch.cat(chunks)
    return chunks[torch.randperm(len(chunks), generator=torch.Generator().manual_seed(0))]


def eval_chunks(tokenizer, seq_len: int, split: str = "test") -> torch.Tensor:
    return token_chunks(tokenizer, _wikitext(split), seq_len)
