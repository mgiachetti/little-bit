import pytest
import torch
import torch.nn.functional as F

from littlebit import bpw_report, convert_model, experts as ex
from littlebit.experts import LittleBitExperts, LittleBitGroupedLinear
from littlebit.layer import dual_svid

transformers = pytest.importorskip("transformers")
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402


def tiny_moe(num_experts=8, top_k=2):
    cfg = Qwen3MoeConfig(vocab_size=128, hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         num_experts=num_experts, num_experts_per_tok=top_k, max_position_embeddings=64)
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(cfg).eval()


def reference_experts(lb: LittleBitExperts, x, idx, w):
    """Token-by-token MoE with the effective (binarized) weights of every expert."""
    gu, dn = lb.gate_up.effective_weight(), lb.down.effective_weight()
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for e, p in zip(idx[t].tolist(), w[t]):
            gate, up = (gu[e] @ x[t]).chunk(2)
            out[t] += p * (dn[e] @ (F.silu(gate) * up))
    return out


def test_batched_init_matches_per_expert():
    torch.manual_seed(0)
    w = torch.randn(4, 48, 32)
    U, V, h, g, l = dual_svid(w, 8, "svid")
    for e in range(4):
        u, v, he, ge, le = dual_svid(w[e], 8, "svid")
        w_batched = (h[e, :, None] * torch.sign(U[e]) * l[e]) @ (torch.sign(V[e]) * g[e, :, None]).T
        w_single = (he[:, None] * torch.sign(u) * le) @ (torch.sign(v) * ge[:, None]).T
        assert torch.allclose(w_batched, w_single, atol=1e-4)


@pytest.mark.parametrize("init", ["svid", "itq"])
def test_experts_forward_matches_reference(init):
    model = tiny_moe()
    block = model.model.layers[0].mlp
    lb = LittleBitExperts(block.experts, r_gate_up=6, r_down=5, init=init)
    torch.manual_seed(1)
    x = torch.randn(10, 64)
    _, w, idx = block.gate(x)
    assert torch.allclose(lb(x, idx, w), reference_experts(lb, x, idx, w), atol=1e-4)


def test_grouped_linear_residual_reduces_error():
    torch.manual_seed(0)
    w = torch.randn(4, 48, 32)
    err = {res: (LittleBitGroupedLinear(w, 8, residual=res).effective_weight() - w).norm()
           for res in (False, True)}
    assert err[True] < err[False]


def test_grouped_and_loop_give_same_grads(monkeypatch):
    model = tiny_moe()
    block = model.model.layers[0].mlp
    torch.manual_seed(1)
    x = torch.randn(12, 64)
    with torch.no_grad():
        _, w, idx = block.gate(x)
    grads = []
    for impl in (ex._grouped_mm, None):  # native/transformers grouped GEMM vs. plain per-expert loop
        monkeypatch.setattr(ex, "_grouped_mm", impl)
        torch.manual_seed(0)
        lb = LittleBitExperts(block.experts, 6, 5, init="itq")
        lb(x, idx, w).pow(2).sum().backward()
        grads.append({n: p.grad.clone() for n, p in lb.named_parameters()})
        assert all(g.abs().sum() > 0 for g in grads[-1].values())
    for n in grads[0]:
        assert torch.allclose(grads[0][n], grads[1][n], rtol=1e-4, atol=1e-5), n


def test_convert_moe_model_and_reload():
    model = tiny_moe()
    ids = torch.randint(0, 128, (2, 16))
    convert_model(model, 0.55, kv_rank_mult=1, init="itq", verbose=False)
    assert all(isinstance(layer.mlp.experts, LittleBitExperts) for layer in model.model.layers)
    rep = bpw_report(model)
    assert rep["converted_params"] > 0 and rep["body_bpw"] == rep["layers_bpw"]  # nothing left in FP
    logits = model(ids).logits
    empty = tiny_moe()
    convert_model(empty, 0.55, kv_rank_mult=1, init="empty", verbose=False)
    empty.load_state_dict(model.state_dict())
    assert torch.allclose(empty(ids).logits, logits, atol=1e-5)
