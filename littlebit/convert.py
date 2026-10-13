"""Swap the nn.Linear layers inside transformer blocks for LittleBitLinear."""

import torch.nn as nn

from .layer import LittleBitLinear, rank_for_bpw

# lm_head and embeddings stay in full precision (paper, Appendix I).
SKIP = ("lm_head", "embed", "wte", "wpe", "project_in", "project_out")
KV_NAMES = ("k_proj", "v_proj")


def _is_gqa_kv(model: nn.Module, name: str, linear: nn.Linear) -> bool:
    """K/V projection narrower than its sibling Q projection (GQA/MQA)."""
    parent_name, _, child = name.rpartition(".")
    if child not in KV_NAMES:
        return False
    q = getattr(model.get_submodule(parent_name), "q_proj", None)
    return isinstance(q, nn.Module) and linear.out_features < q.out_features


def convert_model(model: nn.Module, bpw: float, residual: bool = True,
                  kv_rank_mult: float = 4.0, k: float = 100.0, init: str = "svid",
                  keep: tuple[str, ...] = (), verbose: bool = True) -> nn.Module:
    """Replace every eligible nn.Linear in-place. Returns the model.

    kv_rank_mult: rank multiplier for K/V projections in GQA/MQA models (Appendix B.3).
    init: "svid" (LittleBit), "rot" or "itq" (LittleBit-2).
    keep: module-name suffixes left in full precision (e.g. "layers.2.mlp.down_proj").
    """
    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and not any(s in n for s in SKIP)
               and not any(n.endswith(k) for k in keep)]
    for i, (name, linear) in enumerate(targets):
        r = rank_for_bpw(bpw, linear.out_features, linear.in_features, residual)
        if _is_gqa_kv(model, name, linear):
            r = round(r * kv_rank_mult)
        r = min(r, min(linear.out_features, linear.in_features))
        new = LittleBitLinear(linear, r, residual=residual, k=k, init=init).to(linear.weight.device)
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child, new)
        if verbose:
            print(f"\r[convert] {i + 1}/{len(targets)} {name} r={r}", end="", flush=True)
    if verbose:
        print()
    return model


def bpw_report(model: nn.Module) -> dict:
    """Effective BPW over the converted layers, and total model size in MB."""
    lb_bits = lb_params = other_params = kept_params = 0
    for m in model.modules():
        if isinstance(m, LittleBitLinear):
            lb_bits += m.storage_bits()
            lb_params += m.in_features * m.out_features
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and not any(s in n for s in SKIP):
            kept_params += m.weight.numel()  # body linears left in full precision
    lb_param_ids = {id(p) for m in model.modules() if isinstance(m, LittleBitLinear)
                    for n, p in m.named_parameters() if not n.endswith("bias")}
    seen = set()
    for p in model.parameters():
        if id(p) in lb_param_ids or id(p) in seen:
            continue
        seen.add(id(p))  # tied weights (e.g. lm_head = embed) counted once
        other_params += p.numel()
    total_mb = (lb_bits + 16 * other_params) / 8 / 2**20
    return {
        "layers_bpw": lb_bits / max(lb_params, 1),
        "body_bpw": (lb_bits + 16 * kept_params) / max(lb_params + kept_params, 1),
        "converted_params": lb_params,
        "fp16_params": other_params,
        "size_mb": total_mb,
    }
