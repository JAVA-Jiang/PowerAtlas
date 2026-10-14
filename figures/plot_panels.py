"""Figure 5 bottom-row panels, drawn 1:1 at their final printed size.

Four narrow panels, all from plot_data/falcon_stage_metrics.csv (40 held-out
ECBench instances x 5 training stages of the Falcon3-3B backbone: the untrained
base, the SFT checkpoint, and FA-GRPO steps 20/40/60):

  fig5_traj.pdf    training dynamics -- per-instance grid violation (box) and
                   value-weighted satisfaction (line) over the training stages
  fig5_tierstage.pdf  tier satisfaction S_H / S_M / S_L over the same stages
  fig5_robust.pdf  robustness across renewable penetration and season
  fig5_paired.pdf  per-instance base vs + PowerAtlas, with the y = x diagonal

Everything is a point estimate over the 40 held-out instances; the boxes show
the actual spread across those instances, so no error bars are invented.
Panels are drawn at the width they occupy in the paper (0.19\\textwidth of a
7.0 in text block = 1.33 in) so the type size in the file is the type size on
the page -- no downscaling.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- palette
RED = "#c82423"      # + PowerAtlas / trained  (the eye anchor, as in fig5_1)
BLUE = "#2878b5"     # base / untrained
GREEN = "#54b345"    # third series
AMBER = "#f8ac8c"
GREY = "#8f9299"
GRID = "#DCE1E8"

STAGES = ["Base", "SFT", "GRPO-20", "GRPO-40", "GRPO-60"]
STAGE_LBL = ["Base", "SFT", "G20", "G40", "G60"]
PANEL_W, PANEL_H = 1.33, 1.24          # inches, = 0.19\textwidth of 7.0 in
WIDE_W, WIDE_H = 3.70, 1.86            # inches, = 0.485\textwidth (top-row slot)


def style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 6.90,
        "axes.labelsize": 7.13,
        "axes.titlesize": 7.36,
        "xtick.labelsize": 6.44,
        "ytick.labelsize": 6.44,
        "legend.fontsize": 5.98,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.edgecolor": "#9CA3AF",
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def load(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def col(rows: list[dict], stage: str, key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows if r["stage"] == stage])


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.012)
    fig.savefig(out_dir / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)
    print("wrote", out_dir / f"{name}.pdf")


# ------------------------------------------------------------------ panels
def panel_trajectory(rows, out_dir: Path) -> None:
    """Grid violation (box, left) and S_val (line, right) over training stages."""
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    viol = [col(rows, s, "balance_viol_mw") / 1000.0 for s in STAGES]   # plotted in kMW
    x = np.arange(len(STAGES))

    bp = ax.boxplot(viol, positions=x, widths=0.56, patch_artist=True,
                    showfliers=False, medianprops=dict(color="white", lw=1.0),
                    whiskerprops=dict(color=GREY, lw=0.6),
                    capprops=dict(color=GREY, lw=0.6))
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=RED if i >= 3 else BLUE, edgecolor="none",
                alpha=0.95 if i >= 3 else 0.55)

    ax.set_ylabel("Viol (MWh)", labelpad=1.5)
    ax.set_ylim(0, 24)
    ax.set_yticks([0, 8, 16, 24])
    ax.set_yticklabels(["0", "8k", "16k", "24k"])
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LBL, rotation=30, ha="right")
    ax.tick_params(axis="x", pad=0.6)
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.9)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    sval = [col(rows, s, "s_val").mean() for s in STAGES]
    ax2.plot(x, sval, color="#1f1f1f", marker="o", lw=1.5, ms=3.0,
             mfc="white", mew=1.1, zorder=5, clip_on=False)
    ax2.set_ylabel(r"$S_{val}$ (%)", labelpad=1.5)
    ax2.set_ylim(-4, 104)
    ax2.set_yticks([0, 50, 100])
    ax2.tick_params(axis="y", length=2.0)

    # print the satisfaction the line carries, and the violation it starts and ends at
    for xi, v in zip(x, sval):
        off, ha = ((7.0, 0.0), "left") if xi == 0 else ((0.0, 4.2), "center")
        ax2.annotate(f"{v:.1f}", xy=(xi, v), xytext=off,
                     textcoords="offset points", ha=ha,
                     va="center" if xi == 0 else "bottom",
                     fontsize=4.8, color="#1f1f1f", zorder=6)
    med = [float(np.median(col(rows, s, "balance_viol_mw"))) / 1000.0 for s in STAGES]
    for xi in (0, 3):
        ax.annotate(f"{med[xi]:.1f}k", xy=(xi, med[xi]), xytext=(0, 3.2),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=4.8, fontweight="bold",
                    color=RED if xi >= 3 else BLUE, zorder=6)

    # the checkpoint we ship
    ax.axvspan(2.6, 3.4, color=RED, alpha=0.07, zorder=0)
    tick = ax.get_xticklabels()[3]
    tick.set_color(RED)
    tick.set_fontweight("bold")
    save(fig, out_dir, "fig5_traj")


def panel_tier_stage(rows, out_dir: Path) -> None:
    """S_H / S_M / S_L over the trained stages (the base collapse is in (e))."""
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    trained = STAGES[1:]                       # SFT .. GRPO-60
    lbl_t = STAGE_LBL[1:]
    x = np.arange(len(trained))
    series = [("$S_H$", "s_h", RED, "o", "-"),
              ("$S_M$", "s_m", BLUE, "s", "--"),
              ("$S_L$", "s_l", GREEN, "^", ":")]
    for lbl, key, c, mk, ls in series:
        y = [col(rows, s, key).mean() for s in trained]
        ax.plot(x, y, color=c, marker=mk, ls=ls, label=lbl, lw=1.5, ms=3.2,
                mfc="white", mew=1.1, clip_on=False, zorder=4)
        # S_M is the tier that actually moves, so it carries a value at every
        # checkpoint; the two saturated tiers are labelled at the shipped step
        pts = range(len(trained)) if key == "s_m" else [2]
        for i in pts:
            above = key == "s_l"
            ax.annotate(f"{y[i]:.1f}", xy=(x[i], y[i]),
                        xytext=(0, 4.0 if above else -5.6),
                        textcoords="offset points", ha="center",
                        va="bottom" if above else "top",
                        fontsize=4.7, color=c, zorder=6)

    ax.set_ylabel("Satisfaction (%)", labelpad=1.5)
    ax.set_ylim(83, 108)
    ax.set_yticks([85, 95, 100])
    ax.set_xlim(-0.35, len(trained) - 0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(lbl_t, rotation=30, ha="right")
    ax.tick_params(axis="x", pad=0.6)
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, handlelength=1.1,
              handletextpad=0.3, labelspacing=0.12, borderaxespad=0.1,
              ncol=3, columnspacing=0.5)
    save(fig, out_dir, "fig5_tierstage")


def panel_robust(rows, out_dir: Path) -> None:
    """S_val across renewable penetration and across seasons, base vs + PA."""
    fig, ax = plt.subplots(figsize=(PANEL_W * 1.34, PANEL_H))
    rho = ["1", "2", "4", "6"]
    seasons = ["Winter", "Spring", "Summer I", "Summer II", "Autumn"]
    slab = ["Wi", "Sp", "S1", "S2", "Au"]

    def mean_by(stage, key, val):
        v = [float(r["s_val"]) for r in rows if r["stage"] == stage and r[key] == val]
        return float(np.mean(v))

    base = [mean_by("Base", "penetration_pct", p) for p in rho] + \
           [mean_by("Base", "scenario", s) for s in seasons]
    pa = [mean_by("GRPO-40", "penetration_pct", p) for p in rho] + \
         [mean_by("GRPO-40", "scenario", s) for s in seasons]
    labels = [rf"$\rho${p}" for p in rho] + slab

    x = np.arange(len(labels))
    w = 0.40
    ax.bar(x - w / 2, base, w, color=BLUE, alpha=0.60, label="Base", zorder=3)
    ax.bar(x + w / 2, pa, w, color=RED, alpha=0.95, label="+ PowerAtlas", zorder=3)

    for xi, v in zip(x, pa):
        ax.text(xi + w / 2, v + 2.0, f"{v:.0f}", ha="center", va="bottom",
                fontsize=4.4, color=RED, fontweight="bold")
    for xi, v in zip(x, base):
        if v >= 1.0:
            ax.text(xi - w / 2, v + 2.0, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=4.4, color=BLUE)
    for xi, v in zip(x, base):
        if v < 1.0:
            ax.text(xi - w / 2, 2.0, "0", ha="center", va="bottom", fontsize=5.29,
                    color=BLUE, fontweight="bold")

    # Two readers of an earlier draft merged the two blocks and read the four non-zero base bars as
    # one group, so the split is now a solid rule plus a tinted band, and each block carries its own
    # mean -- the same 40 instances grouped twice must average to the same value.
    ax.axvspan(3.5, 8.6, color=GREY, alpha=0.07, zorder=1)
    ax.axvline(3.5, color=GREY, lw=1.1, zorder=2)
    ax.text(1.5, 106, "by penetration (base 6.8)", ha="center", fontsize=5.0, color=GREY)
    ax.text(6.0, 106, "by season (base 6.8)", ha="center", fontsize=5.0, color=GREY)

    ax.set_ylabel(r"$S_{val}$ (%)", labelpad=1.5)
    ax.set_ylim(0, 132)
    ax.set_yticks([0, 50, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.tick_params(axis="x", pad=0.6)
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", frameon=False, handlelength=0.9,
              handletextpad=0.3, labelspacing=0.12, borderaxespad=0.0,
              ncol=2, columnspacing=0.8, bbox_to_anchor=(0.5, 1.02))
    save(fig, out_dir, "fig5_robust")


def panel_paired(rows, out_dir: Path) -> None:
    """Every held-out instance, before and after, side by side.

    The 40 test instances are sorted by how bad the untrained backbone is on
    them; the blue curve is that backbone, the red curve the shipped checkpoint
    on the *same* instance. The shaded gap between the curves is the reduction,
    so the panel reads left to right without any diagonal to decode.
    """
    fig, ax = plt.subplots(figsize=(PANEL_W * 1.05, PANEL_H))
    ids = [r["instance_id"] for r in rows if r["stage"] == "Base"]
    b = {r["instance_id"]: r for r in rows if r["stage"] == "Base"}
    g = {r["instance_id"]: r for r in rows if r["stage"] == "GRPO-40"}

    pairs = sorted(((float(b[i]["balance_viol_mw"]),
                     float(g[i]["balance_viol_mw"])) for i in ids), reverse=True)
    vb = np.array([p[0] for p in pairs]) / 1000.0
    vp = np.array([p[1] for p in pairs]) / 1000.0
    x = np.arange(1, len(vb) + 1)

    ax.fill_between(x, vp, vb, where=vb >= vp, color=RED, alpha=0.13, lw=0, zorder=2)
    ax.plot(x, vb, color=BLUE, lw=1.4, ls=(0, (3.5, 1.4)), zorder=4, label="base")
    ax.plot(x, vp, color=RED, lw=1.6, zorder=5, label="+ PowerAtlas")
    ax.fill_between(x, 0, vp, color=RED, alpha=0.18, lw=0, zorder=3)

    ax.set_xlabel("Test instance (sorted)", labelpad=1.5)
    ax.set_ylabel("Viol (MWh)", labelpad=1.5)
    ax.set_xlim(1, len(vb))
    ax.set_xticks([1, 10, 20, 30, 40])
    ax.set_ylim(0, 25)
    ax.set_yticks([0, 8, 16, 24])
    ax.set_yticklabels(["0", "8k", "16k", "24k"])
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    ax.annotate("lower on 37/40", xy=(0.50, 0.62), xycoords="axes fraction",
                ha="center", fontsize=5.0, fontweight="bold", color=RED)
    ax.legend(loc="upper right", frameon=False, handlelength=1.2,
              handletextpad=0.3, labelspacing=0.12, borderaxespad=0.15)
    save(fig, out_dir, "fig5_paired")


def panel_temporal(series_path: Path, out_dir: Path) -> None:
    """The two single-instance traces, stacked into one narrow panel.

    Top: hourly grid violation. Bottom: cumulative task violations. Series come
    from plot_data/temporal_series.json, cached by plot_fig_temporal.py (the
    hourly DC power-flow feasibility LP over the real model outputs).
    """
    import json
    from matplotlib.patches import Rectangle
    blob = json.loads(series_path.read_text(encoding="utf-8"))
    colour = {"Qwen3-4B": "#6f5eb5", "Llama-3.2-3B": "#159a8a", "Falcon3-3B": "#d98c3f"}
    dash = {"Qwen3-4B": (0, (4.0, 1.6)),
            "Llama-3.2-3B": (0, (1.4, 1.2)),
            "Falcon3-3B": (0, (5.0, 1.4, 1.0, 1.4))}
    mark = {"Qwen3-4B": "o", "Llama-3.2-3B": "s", "Falcon3-3B": "^"}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(WIDE_W, WIDE_H), sharex=True,
        gridspec_kw=dict(height_ratios=[1.22, 1.0], hspace=0.13))

    # the two windows where this instance is hard: the morning ramp and the
    # evening peak (every non-zero violation hour falls inside one of them)
    for lo, hi in ((5.6, 7.4), (16.6, 22.4)):
        for a in (ax1, ax2):
            a.axvspan(lo, hi, color="#F2C94C", alpha=0.13, lw=0, zorder=0)
    ax1.text(6.5, 0.955, "morning\nramp", transform=ax1.get_xaxis_transform(),
             ha="center", va="top", fontsize=4.8, color="#9A7B1F", linespacing=0.95)
    ax1.text(19.5, 0.955, "evening peak", transform=ax1.get_xaxis_transform(),
             ha="center", va="top", fontsize=4.8, color="#9A7B1F")

    order, entries = ["Qwen3-4B", "Llama-3.2-3B", "Falcon3-3B"], []
    byname = {(s["model"], s["pa"]): s for s in blob["series"]}
    for name in order:
        for pa in (False, True):
            s = byname[(name, pa)]
            c = colour[name]
            x = np.arange(1, len(s["viol_h"]) + 1)
            y = np.asarray(s["viol_h"], float)
            if pa:
                ax1.fill_between(x, 0, y, color=c, alpha=0.22, lw=0, zorder=3)
                ax1.plot(x, y, color=c, lw=1.6, ls="-", marker=mark[name], ms=2.4,
                         markevery=3, mfc="white", mew=0.7, zorder=6)
                ax2.step(x, s["cum"], where="post", color=c, lw=1.6, zorder=6)
            else:
                ax1.plot(x, y, color=c, lw=1.15, ls=dash[name], alpha=0.9, zorder=4)
                ax2.step(x, s["cum"], where="post", color=c, lw=1.15,
                         ls=dash[name], alpha=0.9, zorder=4)
            entries.append((c, pa, name, y.sum(), s["cum"][-1]))

    ax1.set_ylabel("Viol (MWh)", labelpad=1.5, fontsize=6.67)
    ax2.set_ylabel("Task viol", labelpad=1.5, fontsize=6.67)
    ax2.set_xlabel("Hour of day", labelpad=1.5)
    ax2.set_xlim(1, 24)
    ax2.set_xticks([4, 8, 12, 16, 20, 24])
    ax1.set_ylim(0, 700)
    ax1.set_yticks([0, 200, 400, 600])
    ax2.set_ylim(0, 30)
    ax2.set_yticks([0, 12, 24])
    for a in (ax1, ax2):
        a.grid(axis="y", color=GRID, lw=0.5, alpha=0.9)
        a.set_axisbelow(True)
        a.tick_params(labelsize=5.2, pad=0.6)
    ax1.tick_params(axis="x", length=0)

    # the end-of-day task-violation count, printed at the right edge; labels are
    # pushed apart so the near-equal counts stay readable
    placed: list[float] = []
    for c, pa, name, _, cum in sorted(entries, key=lambda e: -e[4]):
        y = float(cum)
        while any(abs(y - q) < 2.6 for q in placed):
            y -= 1.0
        placed.append(y)
        ax2.annotate(f"{cum:.0f}", xy=(24.3, y), color=c, fontsize=5.0,
                     fontweight="bold" if pa else "normal",
                     ha="left", va="center", annotation_clip=False)

    # one legend entry per line -- six lines, six labels, each with the day's
    # total violation (MW.h) so the panel carries its own numbers
    handles, labels = [], []
    for c, pa, name, tot, _ in entries:
        handles.append(plt.Line2D(
            [0], [0], color=c, lw=1.6 if pa else 1.15,
            ls="-" if pa else dash[name],
            marker=mark[name] if pa else "None", ms=2.4, mfc="white", mew=0.7))
        labels.append(f"{name} {'+ PowerAtlas' if pa else 'base'}")
    ax1.legend(handles, labels, loc="lower left", frameon=False, fontsize=5.5,
               handlelength=1.5, handletextpad=0.3, labelspacing=0.10,
               borderaxespad=0.0, ncol=3, columnspacing=0.7,
               bbox_to_anchor=(-0.015, 1.005))
    save(fig, out_dir, "fig5_cd")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("plot_data/falcon_stage_metrics.csv"))
    ap.add_argument("--series", type=Path, default=Path("plot_data/temporal_series.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("figure"))
    args = ap.parse_args()

    style()
    rows = load(args.csv)
    panel_trajectory(rows, args.out_dir)
    panel_tier_stage(rows, args.out_dir)
    panel_robust(rows, args.out_dir)
    panel_paired(rows, args.out_dir)
    if args.series.exists():
        panel_temporal(args.series, args.out_dir)


if __name__ == "__main__":
    main()
