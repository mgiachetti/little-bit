"""Build the report figures from the training logs in results/logs/.

    python scripts/make_figures.py
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOGS, OUT = ROOT / "results" / "logs", ROOT / "results" / "figures"

# Validated categorical slots (dataviz reference palette, light mode) and text/surface tokens.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def evals(log: str) -> list[tuple[int, float]]:
    return [(int(s), float(p)) for s, p in re.findall(r"^step (\d+)\s+eval PPL ([\d.]+)", (LOGS / log).read_text(), re.M)]


def final(log: str) -> float:
    return float(re.search(r"final WikiText-2 PPL: ([\d.]+)", (LOGS / log).read_text()).group(1))


def teacher_line(ax, y, label):
    ax.axhline(y, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate(label, (1, y), xycoords=("axes fraction", "data"), xytext=(-4, 5),
                textcoords="offset points", ha="right", va="bottom", color=INK2, fontsize=10)


def fig_qwen_long():
    pts = evals("qwen_c4.log")
    tokens = [s * 4096 / 1e6 for s, _ in pts]  # 2048 tokens × batch 1 × grad-accum 2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(tokens, [p for _, p in pts], color=BLUE, lw=2, marker="o", ms=5, zorder=3)
    ax.set_yscale("log")
    ax.set_yticks([20, 30, 50, 100, 200, 300])
    ax.set_ylim(17, 330)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    teacher_line(ax, 20.27, "Qwen3-0.6B full precision (20.27)")
    last_t, last_p = tokens[-1], pts[-1][1]
    ax.annotate(f"{last_p:.2f}", (last_t, last_p), xytext=(0, 8), textcoords="offset points",
                ha="center", color=INK, fontsize=10)
    ax.set_xlabel("Training tokens (millions)")
    ax.set_ylabel("WikiText-2 PPL (first 20 windows, log)")
    ax.set_title("Qwen3-0.6B at 0.55 BPW: PPL during the 107M-token run")
    fig.tight_layout()
    fig.savefig(OUT / "qwen3_long_run.png", dpi=150)


def fig_qwen_loss_ablation():
    rows = [("Paper loss (λ·MSE)", final("ab_paper.log")),
            ("Relative MSE", final("ab_msenorm.log")),
            ("Skip token 0", final("ab_skip1.log")),
            ("Skip token 0 + relative MSE", final("ab_skip1_norm.log"))]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = range(len(rows))[::-1]
    ax.barh(list(y), [v for _, v in rows], color=BLUE, height=0.6, zorder=3)
    ax.set_yticks(list(y), [n for n, _ in rows], color=INK)
    for yi, (_, v) in zip(y, rows):
        ax.annotate(f"{v:,.0f}", (v, yi), xytext=(4, 0), textcoords="offset points", va="center", color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(v for _, v in rows) * 1.12)
    ax.set_xlabel("WikiText-2 PPL after 300 steps (full test, lower is better)")
    ax.set_title("Qwen3-0.6B: distillation-loss variants, 300 steps")
    fig.tight_layout()
    fig.savefig(OUT / "qwen3_loss_ablation.png", dpi=150)


def fig_opt_init():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for log, color, name in [("opt125m-bpw0.55-svid.log", ORANGE, "LittleBit (Dual-SVID)"),
                             ("opt125m-bpw0.55-itq.log", BLUE, "LittleBit-2 (Joint-ITQ)")]:
        pts = evals(log)
        ax.plot([s for s, _ in pts], [p for _, p in pts], color=color, lw=2, marker="o", ms=5, label=name, zorder=3)
        ax.annotate(name, pts[-1], xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=10)
    teacher_line(ax, 29.52, "OPT-125M full precision (29.52)")
    ax.set_ylim(0, 330)
    ax.set_xlim(0, 2450)
    ax.set_xlabel("Training step (4 × 512 tokens)")
    ax.set_ylabel("WikiText-2 PPL (first 20 windows)")
    ax.set_title("OPT-125M at 0.55 BPW: initialization, same training recipe")
    ax.legend(loc="upper right", frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "opt125m_init.png", dpi=150)


def fig_qwen_tokens():
    pts = [(300 * 2048 / 1e6, final("ab_skip1_norm.log")),
           (2000 * 2048 / 1e6, final("qwen_long.log")),
           (26000 * 4096 / 1e6, final("qwen_c4.log"))]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([t for t, _ in pts], [p for _, p in pts], color=BLUE, lw=2, marker="o", ms=7, zorder=3)
    for t, p in pts:
        ax.annotate(f"{p:.1f}", (t, p), xytext=(8, 4), textcoords="offset points", color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks([20, 50, 100, 200, 500, 1000])
    ax.set_ylim(17, 1100)
    ax.minorticks_off()
    teacher_line(ax, 20.95, "full precision (20.95)")
    ax.set_xlabel("Training tokens (millions, log)")
    ax.set_ylabel("WikiText-2 PPL (full test, log)")
    ax.set_title("Qwen3-0.6B at 0.55 BPW: final PPL vs. training tokens")
    fig.tight_layout()
    fig.savefig(OUT / "qwen3_tokens.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_qwen_long()
    fig_qwen_loss_ablation()
    fig_opt_init()
    fig_qwen_tokens()
    print("\n".join(str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.png"))))
