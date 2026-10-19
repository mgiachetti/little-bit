"""LittleBit for Mixture-of-Experts layers, with all experts in one grouped GEMM.

transformers stores the experts of a MoE block (Qwen3-MoE, ...) as 3D tensors
gate_up_proj [E, 2I, H] and down_proj [E, H, I]. Each expert matrix gets the same
binarized low-rank factorization as LittleBitLinear, but the latents are stacked
(U [E, d_out, r], V [E, d_in, r], ...) so that sign() runs once for all experts and the
matmuls of all experts run as one grouped GEMM over tokens sorted by expert.

The scales are folded into the binary factors, so a path costs two grouped matmuls:
    x·diag(g)·sign(V)·diag(ℓ)·sign(U)ᵀ·diag(h) = x · [g ⊙ sign(V)] · [h ⊙ sign(U) ⊙ ℓ]ᵀ
and the primary and residual paths are concatenated along the rank, so both paths
together are still two grouped matmuls.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import dual_svid, layer_bits, smooth_sign

try:  # native grouped GEMM on sm80+ (and recent CPU builds), per-expert loop with autograd otherwise
    from transformers.integrations.moe import _grouped_mm
except ImportError:  # pragma: no cover
    _grouped_mm = None


def grouped_mm(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """x [S, d_in] sorted by expert, w [E, d_in, d_out], offs = cumulative tokens per expert."""
    if torch.is_autocast_enabled(x.device.type):  # grouped_mm is not autocast-aware
        x = x.to(torch.get_autocast_dtype(x.device.type))
    if _grouped_mm is not None and x.device.type != "mps":
        return _grouped_mm(x, w.to(x.dtype), offs)
    out, start = [], 0
    for e, end in enumerate(offs.tolist()):
        out.append(x[start:end] @ w[e].to(x.dtype))
        start = end
    return torch.cat(out)


class _GroupedPath(nn.Module):
    """One binarized low-rank path for E stacked experts."""

    def __init__(self, w: torch.Tensor, r: int, k: float, init: str = "svid", chunk: int = 16):
        super().__init__()
        # Batched SVDs of all experts at once need several GB of workspace; do `chunk` experts at a time.
        parts = [dual_svid(w[i:i + chunk], r, init) for i in range(0, w.shape[0], chunk)]
        U, V, h, g, l = (torch.cat(t) for t in zip(*parts))
        self.U = nn.Parameter(U)  # [E, d_out, r]
        self.V = nn.Parameter(V)  # [E, d_in, r]
        self.h = nn.Parameter(h)  # [E, d_out]
        self.g = nn.Parameter(g)  # [E, d_in]
        self.l = nn.Parameter(l)  # [E, r]
        self.k = k

    def factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled binary factors: A = g ⊙ sign(V) [E, d_in, r], B = (h ⊙ sign(U) ⊙ ℓ)ᵀ [E, r, d_out]."""
        a = smooth_sign(self.V, self.k) * self.g[..., None]
        b = smooth_sign(self.U, self.k) * self.h[..., None] * self.l[..., None, :]
        return a, b.mT

    @torch.no_grad()
    def effective_weight(self) -> torch.Tensor:
        u = torch.where(self.U >= 0, 1.0, -1.0)
        v = torch.where(self.V >= 0, 1.0, -1.0)
        return (self.h[..., None] * u * self.l[..., None, :]) @ (v * self.g[..., None]).mT


class LittleBitGroupedLinear(nn.Module):
    """E stacked expert matrices W [E, d_out, d_in] as primary + residual LittleBit paths."""

    def __init__(self, w: torch.Tensor, r: int, residual: bool = True, k: float = 100.0,
                 init: str = "svid"):
        super().__init__()
        self.num_experts, self.out_features, self.in_features = w.shape
        self.r = r
        w = w.detach().float()
        self.primary = _GroupedPath(w, r, k, init)
        self.residual = None
        if residual:
            self.residual = _GroupedPath(w - self.primary.effective_weight(), r, k, init)

    def paths(self):
        return [self.primary] + ([self.residual] if self.residual is not None else [])

    def forward(self, x: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        a, b = zip(*(p.factors() for p in self.paths()))
        a, b = torch.cat(a, -1), torch.cat(b, -2)  # both paths share the two grouped matmuls
        pad = -a.shape[-1] % 8  # grouped GEMM kernels need 16-byte strides; zero rank columns are inert
        if pad:
            a, b = F.pad(a, (0, pad)), F.pad(b, (0, 0, 0, pad))
        return grouped_mm(grouped_mm(x, a, offs), b, offs)

    @torch.no_grad()
    def effective_weight(self) -> torch.Tensor:
        return sum(p.effective_weight() for p in self.paths())

    def storage_bits(self) -> int:
        return self.num_experts * layer_bits(self.r, self.out_features, self.in_features,
                                             self.residual is not None)

    def num_weights(self) -> int:
        return self.num_experts * self.out_features * self.in_features

    def extra_repr(self) -> str:
        bpw = self.storage_bits() / self.num_weights()
        return (f"experts={self.num_experts}, in={self.in_features}, out={self.out_features}, "
                f"r={self.r}, residual={self.residual is not None}, bpw={bpw:.3f}")


class LittleBitExperts(nn.Module):
    """Drop-in replacement for a transformers experts module (gate_up_proj / down_proj 3D tensors).

    Same call signature: forward(hidden_states [T, H], top_k_index [T, K], top_k_weights [T, K]).
    gate_up_proj is factorized as one matrix (shared input-side latents for gate and up).
    """

    def __init__(self, experts: nn.Module, r_gate_up: int, r_down: int, residual: bool = True,
                 k: float = 100.0, init: str = "svid"):
        super().__init__()
        self.num_experts = experts.gate_up_proj.shape[0]
        self.act_fn = experts.act_fn
        self.gate_up = LittleBitGroupedLinear(experts.gate_up_proj, r_gate_up, residual, k, init)
        self.down = LittleBitGroupedLinear(experts.down_proj, r_down, residual, k, init)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        top_k = top_k_index.shape[-1]
        flat = top_k_index.reshape(-1)
        order = torch.argsort(flat, stable=True)  # (token, slot) pairs grouped by expert
        token_idx = order // top_k
        counts = torch.bincount(flat, minlength=self.num_experts)
        offs = torch.cumsum(counts, 0).to(torch.int32)
        gate, up = self.gate_up(hidden_states[token_idx], offs).chunk(2, dim=-1)
        out = self.down(self.act_fn(gate) * up, offs)
        out = out * top_k_weights.reshape(-1)[order, None].to(out.dtype)
        final = torch.zeros_like(hidden_states)
        return final.index_add_(0, token_idx, out.to(final.dtype))
