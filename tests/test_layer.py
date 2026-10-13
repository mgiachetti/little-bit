import torch
import torch.nn as nn

from littlebit import LittleBitLinear, convert_model, bpw_report, layer_bits, rank_for_bpw, smooth_sign


def test_rank_matches_paper_appendix_d():
    assert rank_for_bpw(0.55, 4096, 4096) == 546
    assert rank_for_bpw(0.1, 4096, 11008) == 133
    assert abs(layer_bits(546, 4096, 4096) / 4096**2 - 0.5498) < 1e-3
    assert abs(layer_bits(133, 4096, 11008) / (4096 * 11008) - 0.0999) < 1e-3


def test_smooth_sign_forward_and_grad():
    x = torch.tensor([-0.5, -1e-3, 0.0, 1e-3, 0.5], requires_grad=True)
    y = smooth_sign(x, 100.0)
    assert y.tolist() == [-1, -1, 1, 1, 1]
    y.sum().backward()
    expected = 100 * (1 - torch.tanh(100 * x.detach()) ** 2)
    assert torch.allclose(x.grad, expected)


def test_forward_equals_effective_weight():
    torch.manual_seed(0)
    lin = nn.Linear(64, 48)
    lb = LittleBitLinear(lin, r=8)
    x = torch.randn(3, 5, 64)
    ref = x @ lb.effective_weight().T + lb.bias
    assert torch.allclose(lb(x), ref, atol=1e-5)


def test_residual_reduces_init_error():
    torch.manual_seed(0)
    lin = nn.Linear(128, 128, bias=False)
    w = lin.weight.detach()
    err_single = (w - LittleBitLinear(lin, r=16, residual=False).effective_weight()).norm()
    err_res = (w - LittleBitLinear(lin, r=16, residual=True).effective_weight()).norm()
    assert err_res < err_single


def test_gradients_reach_latents_and_scales():
    torch.manual_seed(0)
    lb = LittleBitLinear(nn.Linear(32, 16), r=4)
    lb(torch.randn(2, 32)).pow(2).sum().backward()
    for name, p in lb.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_convert_skips_head_and_boosts_gqa_kv():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(256, 256)
            self.k_proj = nn.Linear(256, 64)

    model = nn.Module()
    model.layers = nn.ModuleList([Block()])
    model.lm_head = nn.Linear(256, 1000)
    convert_model(model, 0.3, verbose=False)
    assert isinstance(model.layers[0].q_proj, LittleBitLinear)
    assert isinstance(model.lm_head, nn.Linear)
    base_r = rank_for_bpw(0.3, 64, 256)
    assert model.layers[0].k_proj.r == round(base_r * 4)
    assert bpw_report(model)["converted_params"] == 256 * 256 + 256 * 64


def _distortion(m: torch.Tensor) -> float:
    """Mean local distortion λ(u) = 1 - ||u||₁² / (r ||u||₂²) over rows (LittleBit-2, Eq. 4)."""
    r = m.shape[1]
    return (1 - m.abs().sum(1) ** 2 / (r * m.pow(2).sum(1))).mean().item()


def test_itq_preserves_w_and_lowers_distortion():
    from littlebit.layer import joint_itq
    torch.manual_seed(0)
    w = torch.randn(96, 32) @ torch.randn(32, 80)
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    U, V = u[:, :16] * s[:16].sqrt(), vh[:16].T * s[:16].sqrt()
    R = joint_itq(U, V)
    assert torch.allclose(R @ R.T, torch.eye(16), atol=1e-4)
    assert torch.allclose((U @ R) @ (V @ R).T, U @ V.T, atol=1e-3)
    Z = torch.cat([U, V])
    assert _distortion(Z @ R) < _distortion(Z)


def test_all_inits_build_a_layer():
    torch.manual_seed(0)
    lin = nn.Linear(64, 48)
    for init in ("svid", "rot", "itq"):
        lb = LittleBitLinear(lin, r=8, init=init)
        assert lb(torch.randn(2, 64)).shape == (2, 48)


def test_empty_init_loads_trained_state():
    torch.manual_seed(0)
    lin = nn.Linear(64, 48)
    trained = LittleBitLinear(lin, r=8, init="itq")
    shell = LittleBitLinear(lin, r=8, init="empty")
    shell.load_state_dict(trained.state_dict())
    x = torch.randn(3, 64)
    assert torch.equal(shell(x), trained(x))
