"""Figure 1 -- cross-model performance landscape (Table 1 + Table 2).

x = value-weighted task satisfaction S_val (%), y = grid violation (MW, log),
bubble size = operating cost ($). Frontier LLMs and open-weight base models are
kept calm; the three + PowerAtlas results are red so they anchor the eye in the
good region (high satisfaction, low violation). Arrows connect each base backbone
to its + PowerAtlas counterpart. Point estimates on the hardest ECBench instance;
no error bars.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Official palette.
TEAL, BLUE, RED, CONNECT = "#45a29b", "#3d7ec2", "#6f5eb5", "#C9CFD6"
CAT_COLOR = {"Frontier LLM": TEAL, "Open-weight base": BLUE, "+ PowerAtlas": RED}
CAT_MARKER = {"Frontier LLM": "o", "Open-weight base": "s", "+ PowerAtlas": "*"}

# (name, category, Viol MW, S_val %, Cost $)  -- verbatim from Tables 1 & 2.
POINTS = [
    ("DeepSeek-V4-Flash",  "Frontier LLM",     5234.2, 23.4, 47948178),
    ("DeepSeek-V4-Pro",    "Frontier LLM",     2763.4, 22.1, 25762501),
    ("Kimi-K2.6-fast",     "Frontier LLM",     1212.0, 55.5, 12392536),
    ("Kimi-K2.6-pro",      "Frontier LLM",     1488.6, 53.4, 14838689),
    ("Qwen3.7-Plus",       "Frontier LLM",      741.0, 83.1, 9344901),
    ("Qwen3.7-Max",        "Frontier LLM",      462.6, 86.2, 5725763),
    ("GPT-5.5",            "Frontier LLM",      445.8, 66.9, 6068833),
    ("Claude-Haiku-4.5",   "Frontier LLM",     543.3, 57.6, 7548683),
    ("Qwen3-4B",           "Open-weight base",  1766.9, 50.9, 17706710),
    ("Llama-3.2-3B",       "Open-weight base",  6492.1, 33.3, 59997272),
    ("Falcon3-3B",         "Open-weight base", 11079.2,  7.4, 101484114),
    ("Qwen3-4B + PA",      "+ PowerAtlas",       274.7, 89.0,  4671682),
    ("Llama-3.2-3B + PA",  "+ PowerAtlas",       282.1, 99.7,  3832071),
    ("Falcon3-3B + PA",    "+ PowerAtlas",       216.6, 94.3,  3928513),
]

# base -> +PA arrows.
PAIRS = [("Qwen3-4B", "Qwen3-4B + PA"), ("Llama-3.2-3B", "Llama-3.2-3B + PA"),
         ("Falcon3-3B", "Falcon3-3B + PA")]

# Full names exactly as in Tables 1 & 2.
FULL = {
    "DeepSeek-V4-Flash": "DeepSeek-V4-Flash", "DeepSeek-V4-Pro": "DeepSeek-V4-Pro",
    "Kimi-K2.6-fast": "Kimi-K2.6 (fast)", "Kimi-K2.6-pro": "Kimi-K2.6 (pro)",
    "Claude-Haiku-4.5": "Claude Haiku 4.5", "Qwen3.7-Plus": "Qwen3.7-Plus",
    "Qwen3.7-Max": "Qwen3.7-Max", "GPT-5.5": "GPT-5.5",
    "Qwen3-4B": "Qwen3-4B-Instruct", "Llama-3.2-3B": "Llama-3.2-3B-Instruct",
    "Falcon3-3B": "Falcon3-3B-Instruct",
    "Qwen3-4B + PA": "Qwen3-4B-Instruct + PowerAtlas",
    "Llama-3.2-3B + PA": "Llama-3.2-3B-Instruct + PowerAtlas",
    "Falcon3-3B + PA": "Falcon3-3B-Instruct + PowerAtlas",
}
# (dx pt, dy pt, ha) for the non-highlighted points (labelled in place).
LABEL_OFF = {
    "DeepSeek-V4-Flash": (-9, 12, "right"), "DeepSeek-V4-Pro": (11, 2, "left"),
    "Kimi-K2.6-fast": (9, 9, "left"), "Kimi-K2.6-pro": (-8, -14, "right"),
    "Qwen3.7-Plus": (10, 7, "left"), "Qwen3.7-Max": (-9, -14, "right"),
    "GPT-5.5": (10, -11, "left"), "Claude-Haiku-4.5": (-10, 11, "right"),
    "Qwen3-4B": (14, 5, "left"), "Llama-3.2-3B": (15, 3, "left"), "Falcon3-3B": (16, 2, "left"),
}
# The three + PowerAtlas points: two-line label (model name / + PowerAtlas) with a
# near-horizontal leader. Llama on the right, Qwen and Falcon on the left.
# key, top-line model name, (anchor x, anchor y in data coords), ha
PA_LABELS = [
    ("Qwen3-4B + PA",    "Qwen3-4B-Instruct",     (72, 345),  "right"),
    ("Falcon3-3B + PA",  "Falcon3-3B-Instruct",   (72, 208),  "right"),
    ("Llama-3.2-3B + PA", "Llama-3.2-3B-Instruct", (107, 300), "left"),
]


def create_figure(out_dir: Path) -> tuple[Path, Path]:
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 18.6, "axes.labelsize": 21.4, "xtick.labelsize": 18.6, "ytick.labelsize": 18.6,
        "axes.edgecolor": "#6B7280", "axes.linewidth": 1.4,
        "axes.labelweight": "bold", "pdf.fonttype": 42, "ps.fonttype": 42,
        # Render math ($S_{val}$ etc.) in Times New Roman too, matching the serif text.
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
    })

    idx = {p[0]: p for p in POINTS}
    costs = np.array([p[4] for p in POINTS], float)
    lo, hi = costs.min(), costs.max()

    def size_of(cost: float) -> float:
        return float(np.interp(cost, (lo, hi), (95, 560)))

    # Height trimmed (6.4 -> 5.4) to shrink the on-page panel WITHOUT changing the
    # effective font: figsize WIDTH and the LaTeX render width are unchanged, so the
    # font/figure-width ratio (hence on-page pt) is identical -- only the aspect gets wider.
    fig, ax = plt.subplots(figsize=(10.8, 5.4))

    # base -> +PA arrows first (under the markers)
    for base, final in PAIRS:
        b, f = idx[base], idx[final]
        ax.annotate("", xy=(f[3], f[2]), xytext=(b[3], b[2]),
                    arrowprops=dict(arrowstyle="-|>", color=CONNECT, lw=2.0,
                                    shrinkA=10, shrinkB=13, connectionstyle="arc3,rad=-0.12"),
                    zorder=1)

    for cat in ["Frontier LLM", "Open-weight base", "+ PowerAtlas"]:
        pts = [p for p in POINTS if p[1] == cat]
        xs = [p[3] for p in pts]
        ys = [p[2] for p in pts]
        ss = [size_of(p[4]) * (1.7 if cat == "+ PowerAtlas" else 1.0) for p in pts]
        ax.scatter(xs, ys, s=ss, c=CAT_COLOR[cat], marker=CAT_MARKER[cat],
                   edgecolor="white", linewidth=1.2, alpha=0.95,
                   zorder=4 if cat == "+ PowerAtlas" else 3, label=cat)

    # in-place labels for frontier + base points
    for name, cat, viol, sval, cost in POINTS:
        if cat == "+ PowerAtlas":
            continue
        dx, dy, ha = LABEL_OFF.get(name, (8, 5, "left"))
        ax.annotate(FULL[name], (sval, viol), xytext=(dx, dy), textcoords="offset points",
                    fontsize=14.6, color="#22303C", ha=ha, va="center")

    # two-line, leader-lined labels for the three + PowerAtlas stars
    for key, top, (tx, ty), ha in PA_LABELS:
        _, _, viol, sval, _ = idx[key]
        ax.annotate(f"{top}\n+ PowerAtlas", xy=(sval, viol), xytext=(tx, ty), textcoords="data",
                    fontsize=14.9, color=RED, fontweight="bold", ha=ha, va="center", linespacing=1.4,
                    arrowprops=dict(arrowstyle="-", color=RED, lw=1.0, alpha=0.6,
                                    shrinkA=4, shrinkB=7))

    ax.set_yscale("log")
    ax.set_xlim(-2, 128)
    ax.set_ylim(140, 20000)
    ax.set_xticks([0, 20, 40, 60, 80, 100])  # satisfaction caps at 100
    ax.set_xlabel(r"Value-weighted task satisfaction  $S_{val}$ (%)")
    ax.set_ylabel(r"Grid violation  (MWh)")
    ax.grid(axis="both", which="major", color="#DCE1E8", linewidth=0.7, alpha=0.85)
    ax.grid(axis="y", which="minor", visible=False)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.legend(loc="upper right", frameon=True, fontsize=17.2, handletextpad=0.5,
              borderpad=0.6, labelspacing=0.5, scatterpoints=1, markerscale=0.8)
    ax.text(0.015, 0.03, "bubble size " + r"$\propto$" + " operating cost",
            transform=ax.transAxes, fontsize=15.9, color="#5B6573", style="italic")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig5_1.png"
    pdf_path = out_dir / "fig5_1.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("figure"))
    args = parser.parse_args()
    png_path, pdf_path = create_figure(args.out_dir)
    print(f"Wrote {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
