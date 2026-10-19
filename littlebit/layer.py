"""LittleBit linear layer (Lee et al., NeurIPS 2025, arXiv 2506.13771).

W ≈ diag(h) · sign(U) · diag(ℓ) · sign(V)ᵀ · diag(g)            (Eq. 4)
plus a parallel residual path with the same structure        (Eq. 9-10).
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _scaled_signs(paths: list[tuple[torch.Tensor, ...]], k: float, dtype: torch.dtype):
    """Concatenated scaled binary factors of all paths, rank padded to a multiple of 8.

    A = [g ⊙ sign(V)]  [.., d_in, R],  B = [h ⊙ sign(U) ⊙ ℓ]ᵀ  [.., R, d_out]  (leading dims: experts).
    Folding the scales here keeps elementwise work off the activations; what is left is
    elementwise work over the latents, compiled on CUDA into a few fused kernels.
    """
    a = torch.cat([smooth_sign(V, k) * g[..., None] for U, V, h, g, l in paths], -1)
    b = torch.cat([smooth_sign(U, k) * h[..., None] * l[..., None, :] for U, V, h, g, l in paths], -1)
    pad = -a.shape[-1] % 8  # grouped GEMM kernels need 16-byte strides; zero rank columns are inert
    if pad:
        a, b = F.pad(a, (0, pad)), F.pad(b, (0, pad))
    return a.to(dtype), b.mT.to(dtype)


_scaled_signs_compiled = torch.compile(_scaled_signs, dynamic=False)
# one specialization per (shape, rank, grad mode): a model has more than dynamo's default 8
torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
_compile_ok = True


def scaled_signs(paths: list[tuple[torch.Tensor, ...]], k: float, dtype: torch.dtype):
    """_scaled_signs, compiled where Triton works (CUDA sm70+), eager otherwise or if compiling fails."""
    global _compile_ok
    x = paths[0][0]
    if _compile_ok and x.is_cuda and torch.cuda.get_device_capability(x.device) >= (7, 0):
        try:
            return _scaled_signs_compiled(paths, k, dtype)
        except Exception as e:  # e.g. no C compiler / Python headers for Triton
            warnings.warn(f"torch.compile unavailable, using eager LittleBit factors: {e!r:.200}")
            _compile_ok = False
    return _scaled_signs(paths, k, dtype)


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


def _rank1_abs(m: torch.Tensor, fast: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Best rank-1 approximation |m| ≈ a·bᵀ with a, b >= 0 (batched over leading dims).

    fast: power iteration instead of a full SVD (the Perron pair of |m| has a wide gap).
    """
    if fast:
        m = m.abs()
        b = m.mean(-2)
        for _ in range(20):
            b = (m.mT @ (m @ b[..., None]))[..., 0]
            b = b / b.norm(dim=-1, keepdim=True)
        a = (m @ b[..., None])[..., 0]
        s = a.norm(dim=-1, keepdim=True)
        return a / s.sqrt(), b * s.sqrt()
    u, s, vh = torch.linalg.svd(m.abs(), full_matrices=False)
    a = u[..., :, 0] * s[..., :1].sqrt()
    b = vh[..., 0, :] * s[..., :1].sqrt()
    # Perron vectors of a non-negative matrix share a sign; make them positive.
    return a.abs(), b.abs()


def _polar(m: torch.Tensor, fast: bool = False) -> torch.Tensor:
    """Orthogonal polar factor of mᵀ: ψφᵀ for m = φ Σ ψᵀ (the Procrustes solution).

    fast: Newton-Schulz iteration (matmuls only) instead of a batched SVD, which cuSOLVER
    runs slowly for many small matrices.
    """
    if fast:
        x = m.mT / m.norm(dim=(-2, -1), keepdim=True)
        for _ in range(30):
            x = 1.5 * x - 0.5 * x @ x.mT @ x
        return x
    phi, _, psi_t = torch.linalg.svd(m, full_matrices=False)
    return psi_t.mT @ phi.mT


def joint_itq(U: torch.Tensor, V: torch.Tensor, n_iter: int = 50, fast: bool = False) -> torch.Tensor:
    """LittleBit-2 Joint-ITQ (arXiv 2603.00042, Algorithm 1).

    Finds an orthogonal R (r×r) aligning Z = [U; V] with the binary hypercube:
    min ||sign(ZR) - ZR||. Since (UR)(VR)ᵀ = UVᵀ, W is unchanged.
    Leading dims are batch dims (e.g. one R per MoE expert).
    """
    Z = torch.cat([U, V], dim=-2)
    R = _random_orthogonal(Z.shape[:-2], U.shape[-1], Z)
    for _ in range(n_iter):
        B = torch.where(Z @ R >= 0, 1.0, -1.0)
        R = _polar(B.mT @ Z, fast)
    if fast:  # exactly orthogonal, so that W = (UR)(VR)ᵀ is unchanged
        R = _polar(R.mT)
    return R


def _random_orthogonal(batch: torch.Size, r: int, like: torch.Tensor) -> torch.Tensor:
    """Random orthogonal r×r matrix, or a batch of them (one per expert)."""
    kw = {"dtype": like.dtype, "device": like.device}
    if not batch:
        return nn.init.orthogonal_(torch.empty(r, r, **kw))
    q, rr = torch.linalg.qr(torch.randn(*batch, r, r, **kw))
    return q * torch.diagonal(rr, dim1=-2, dim2=-1).sign()[..., None, :]


def _top_svd(w: torch.Tensor, r: int, fast: bool = False):
    """Top-r singular triplets of w: (u [.., d_out, r], s [.., r], v [.., d_in, r]).

    fast: eigendecomposition of the smaller Gram matrix instead of a full SVD.
    """
    if not fast:
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        return u[..., :, :r], s[..., :r], vh[..., :r, :].mT
    wide = w.shape[-2] <= w.shape[-1]
    lam, q = torch.linalg.eigh(w @ w.mT if wide else w.mT @ w)
    q, s = q.flip(-1)[..., :r], lam.flip(-1)[..., :r].clamp_min(0).sqrt()
    other = (w.mT @ q if wide else w @ q) / s.clamp_min(1e-12)[..., None, :]
    return (q, s, other) if wide else (other, s, q)


def dual_svid(w: torch.Tensor, r: int, init: str = "svid", itq_iters: int = 50, fast: bool = False):
    """Dual-SVID initialization (Section 3.2, Eqs. 6-8).

    init: "svid" (LittleBit), "rot" (random latent rotation), "itq" (LittleBit-2 Joint-ITQ),
    or "empty" (shapes only, no SVD — for loading a trained checkpoint).
    Returns latent factors U (d_out×r), V (d_in×r) and scales h, g, ℓ.
    w may carry leading batch dims (stacked MoE experts); everything is then per batch entry.
    fast: GPU-friendly linear algebra (Gram eigh, Newton-Schulz, power iteration) instead of
    exact SVDs; same result up to float error, ~10x faster for stacked experts.
    """
    w = w.float()
    if init == "empty":
        *b, d_out, d_in = w.shape
        kw = {"device": w.device}
        return (torch.zeros(*b, d_out, r, **kw), torch.zeros(*b, d_in, r, **kw),
                torch.ones(*b, d_out, **kw), torch.ones(*b, d_in, **kw), torch.ones(*b, r, **kw))
    u, s, v = _top_svd(w, r, fast)
    sq = s[..., None, :].sqrt()
    U = u * sq      # W ≈ U Vᵀ, singular values split evenly
    V = v * sq
    if init == "rot":
        R = _random_orthogonal(w.shape[:-2], r, U)
        U, V = U @ R, V @ R
    elif init == "itq":
        R = joint_itq(U, V, itq_iters, fast)
        U, V = U @ R, V @ R
    elif init != "svid":
        raise ValueError(f"unknown init {init!r}")
    h, l_u = _rank1_abs(U, fast)      # |U| ≈ h ℓ_uᵀ
    g, l_v = _rank1_abs(V, fast)      # |V| ≈ g ℓ_vᵀ
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

    def latents(self) -> tuple[torch.Tensor, ...]:
        return self.U, self.V, self.h, self.g, self.l

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Y = ((((X ⊙ g) V_sign) ⊙ ℓ) U_signᵀ) ⊙ h       (Eq. 5), scales folded into the factors
        a, b = scaled_signs([self.latents()], self.k, x.dtype)
        return x @ a @ b

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

    def paths(self):
        return [self.primary] + ([self.residual] if self.residual is not None else [])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eq. 5 with the scales folded into the binary factors and both paths concatenated along
        # the rank: two matmuls per layer and no elementwise passes over the activations.
        a, b = scaled_signs([p.latents() for p in self.paths()], self.primary.k, x.dtype)
        y = x @ a @ b
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

    def num_weights(self) -> int:
        return self.out_features * self.in_features

    def extra_repr(self) -> str:
        bpw = self.storage_bits() / (self.out_features * self.in_features)
        return (f"in={self.in_features}, out={self.out_features}, r={self.r}, "
                f"residual={self.residual is not None}, bpw={bpw:.3f}")
