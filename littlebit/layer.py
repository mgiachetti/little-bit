"""LittleBit linear layer (Lee et al., NeurIPS 2025, arXiv 2506.13771).

W ≈ diag(h) · sign(U) · diag(ℓ) · sign(V)ᵀ · diag(g)            (Eq. 4)
plus a parallel residual path with the same structure        (Eq. 9-10).
"""

import torch
import torch.nn as nn


class SmoothSign(torch.autograd.Function):
    """Forward: sign(x) in {-1, +1}. Backward: d/dx tanh(k·x) (Appendix B.2)."""

    @staticmethod
    def forward(ctx, x, k):
        ctx.save_for_backward(x)
        ctx.k = k
        return torch.where(x >= 0, 1.0, -1.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        t = torch.tanh(ctx.k * x)
        return grad_out * ctx.k * (1 - t * t), None


def smooth_sign(x: torch.Tensor, k: float = 100.0) -> torch.Tensor:
    return SmoothSign.apply(x, k)


def rank_for_bpw(bpw: float, d_out: int, d_in: int, residual: bool = True) -> int:
    """Latent rank r that hits a target bits-per-weight (Eq. 17, Appendix D)."""
    paths = 2 if residual else 1
    num = bpw * d_out * d_in - paths * 16 * (d_out + d_in)
    den = paths * (d_out + d_in + 16)
    return max(1, round(num / den))


def layer_bits(r: int, d_out: int, d_in: int, residual: bool = True) -> int:
    """Storage bits of one LittleBit layer (Eq. 16 numerator). Bias excluded."""
    paths = 2 if residual else 1
    return paths * (r * (d_out + d_in) + 16 * (d_out + d_in) + 16 * r)


def _rank1_abs(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Best rank-1 approximation |m| ≈ a·bᵀ with a, b >= 0."""
    u, s, vh = torch.linalg.svd(m.abs(), full_matrices=False)
    a = u[:, 0] * s[0].sqrt()
    b = vh[0] * s[0].sqrt()
    # Perron vectors of a non-negative matrix share a sign; make them positive.
    return a.abs(), b.abs()


def joint_itq(U: torch.Tensor, V: torch.Tensor, n_iter: int = 50) -> torch.Tensor:
    """LittleBit-2 Joint-ITQ (arXiv 2603.00042, Algorithm 1).

    Finds an orthogonal R (r×r) aligning Z = [U; V] with the binary hypercube:
    min ||sign(ZR) - ZR||. Since (UR)(VR)ᵀ = UVᵀ, W is unchanged.
    """
    Z = torch.cat([U, V], dim=0)
    R = nn.init.orthogonal_(torch.empty(U.shape[1], U.shape[1], dtype=Z.dtype, device=Z.device))
    for _ in range(n_iter):
        B = torch.where(Z @ R >= 0, 1.0, -1.0)
        phi, _, psi_t = torch.linalg.svd(B.T @ Z, full_matrices=False)
        R = psi_t.T @ phi.T
    return R


def dual_svid(w: torch.Tensor, r: int, init: str = "svid", itq_iters: int = 50):
    """Dual-SVID initialization (Section 3.2, Eqs. 6-8).

    init: "svid" (LittleBit), "rot" (random latent rotation), "itq" (LittleBit-2 Joint-ITQ),
    or "empty" (shapes only, no SVD — for loading a trained checkpoint).
    Returns latent factors U (d_out×r), V (d_in×r) and scales h, g, ℓ.
    """
    w = w.float()
    if init == "empty":
        (d_out, d_in), kw = w.shape, {"device": w.device}
        return (torch.zeros(d_out, r, **kw), torch.zeros(d_in, r, **kw),
                torch.ones(d_out, **kw), torch.ones(d_in, **kw), torch.ones(r, **kw))
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    sq = s[:r].sqrt()
    U = u[:, :r] * sq           # W ≈ U Vᵀ, singular values split evenly
    V = vh[:r].T * sq
    if init == "rot":
        R = nn.init.orthogonal_(torch.empty(r, r, device=w.device))
        U, V = U @ R, V @ R
    elif init == "itq":
        R = joint_itq(U, V, itq_iters)
        U, V = U @ R, V @ R
    elif init != "svid":
        raise ValueError(f"unknown init {init!r}")
    h, l_u = _rank1_abs(U)      # |U| ≈ h ℓ_uᵀ
    g, l_v = _rank1_abs(V)      # |V| ≈ g ℓ_vᵀ
    return U, V, h, g, l_u * l_v


class _Path(nn.Module):
    """One binarized low-rank path: diag(h)·sign(U)·diag(ℓ)·sign(V)ᵀ·diag(g)."""

    def __init__(self, w: torch.Tensor, r: int, k: float, init: str = "svid"):
        super().__init__()
        U, V, h, g, l = dual_svid(w, r, init)
        self.U = nn.Parameter(U)
        self.V = nn.Parameter(V)
        self.h = nn.Parameter(h)
        self.g = nn.Parameter(g)
        self.l = nn.Parameter(l)
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Y = ((((X ⊙ g) V_sign) ⊙ ℓ) U_signᵀ) ⊙ h       (Eq. 5)
        v = smooth_sign(self.V, self.k).to(x.dtype)
        u = smooth_sign(self.U, self.k).to(x.dtype)
        y = (x * self.g.to(x.dtype)) @ v
        y = (y * self.l.to(x.dtype)) @ u.T
        return y * self.h.to(x.dtype)

    @torch.no_grad()
    def effective_weight(self) -> torch.Tensor:
        u = torch.where(self.U >= 0, 1.0, -1.0)
        v = torch.where(self.V >= 0, 1.0, -1.0)
        return (self.h[:, None] * u * self.l) @ (v * self.g[:, None]).T


class LittleBitLinear(nn.Module):
    """Drop-in replacement for nn.Linear with primary + residual LittleBit paths."""

    def __init__(self, linear: nn.Linear, r: int, residual: bool = True, k: float = 100.0,
                 init: str = "svid"):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        w = linear.weight.detach().float()
        self.primary = _Path(w, r, k, init)
        self.residual = None
        if residual:
            w_res = w - self.primary.effective_weight()
            self.residual = _Path(w_res, r, k, init)
        # Bias is kept in full precision (the paper's BPW counts only W).
        self.bias = None
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().float().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.primary(x)
        if self.residual is not None:
            y = y + self.residual(x)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y

    @torch.no_grad()
    def effective_weight(self) -> torch.Tensor:
        w = self.primary.effective_weight()
        if self.residual is not None:
            w = w + self.residual.effective_weight()
        return w

    def storage_bits(self) -> int:
        return layer_bits(self.r, self.out_features, self.in_features, self.residual is not None)

    def extra_repr(self) -> str:
        bpw = self.storage_bits() / (self.out_features * self.in_features)
        return (f"in={self.in_features}, out={self.out_features}, r={self.r}, "
                f"residual={self.residual is not None}, bpw={bpw:.3f}")
