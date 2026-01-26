#!/usr/bin/env python3
"""
plot_he_regularization_results.py

Revisions:
- show p-values from paired t-tests.
- scatter_ood_vs_modHE:
    * per-run arrows (transparent) + per-seed average arrows (bold)
    * bracket-style annotations showing:
        - p-value for vertical metric (OOD) and
        - p-value for horizontal metric (Modifier HE)

- bar_he_summary:
    * t-test p-values displayed (Modifier HE, Sequence HE)

- ood_accuracy_by_complexity:
    * horizontal subplots (one row, columns per noise)

Usage:
  python plot_he_regularization_results.py --json results_he_reg_strong/rolling_results.json --outdir plots_he_reg
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Optional scipy t-test (recommended). If absent, we use a simple fallback approximation.
# ----------------------------
def _try_import_scipy():
    try:
        from scipy import stats  # type: ignore
        return stats
    except Exception:
        return None


STATS = _try_import_scipy()


def paired_ttest_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided paired t-test p-value for arrays a vs b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 2:
        return float("nan")

    if STATS is not None:
        return float(STATS.ttest_rel(b, a, nan_policy="omit").pvalue)

    # Fallback normal approximation on mean difference
    d = b - a
    se = np.std(d, ddof=1) / np.sqrt(len(d))
    if se == 0:
        return 1.0
    t = np.mean(d) / se
    z = abs(t)
    return float(2 * (1 - 0.5 * (1 + np.math.erf(z / np.sqrt(2)))))


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "p=NA"
    if p < 1e-4:
        return "p<1e-4"
    if p < 1e-3:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


# ----------------------------
# Styling
# ----------------------------
def apply_ijcai_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_fig(fig, outdir: str, name: str, dpi: int = 300) -> None:
    png = os.path.join(outdir, f"{name}.png")
    pdf = os.path.join(outdir, f"{name}.pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")


# ----------------------------
# Data helpers
# ----------------------------
def mean_ood_acc(ood_acc: Dict, nmin: int = 5, nmax: int = 12) -> float:
    vals = []
    for n in range(nmin, nmax + 1):
        if str(n) in ood_acc:
            vals.append(float(ood_acc[str(n)]))
        elif n in ood_acc:
            vals.append(float(ood_acc[n]))
    return float(np.mean(vals)) if vals else float("nan")


def mean_layerwise(d: Dict) -> float:
    vals = [float(v) for v in d.values()]
    return float(np.mean(vals)) if vals else float("nan")


def sorted_layers(d: Dict) -> Tuple[np.ndarray, np.ndarray]:
    items = [(int(k), float(v)) for k, v in d.items()]
    items.sort(key=lambda x: x[0])
    xs = np.array([k for k, _ in items], dtype=int)
    ys = np.array([v for _, v in items], dtype=float)
    return xs, ys


def group_runs_by_key(runs: List[Dict[str, Any]]):
    g = {}
    for r in runs:
        cfg = r["config"]
        key = (int(cfg["num_noise"]), int(cfg["seed"]), str(cfg["condition"]))
        g[key] = r
    return g


def unique_sorted_int(runs: List[Dict[str, Any]], field: str) -> List[int]:
    return sorted(list({int(r["config"][field]) for r in runs}))


def seed_level_vectors(runs: List[Dict[str, Any]]):
    """
    Seed-level aggregates (avg across noise levels within each seed).

    Returns:
      seeds, dict of np.arrays:
        ood_baseline, ood_reg
        modhe_baseline, modhe_reg
        seqhe_baseline, seqhe_reg
    """
    g = group_runs_by_key(runs)
    noise_levels = unique_sorted_int(runs, "num_noise")
    seeds = unique_sorted_int(runs, "seed")

    def per_seed(condition: str, getter):
        out = []
        for s in seeds:
            vals = []
            for n in noise_levels:
                r = g.get((n, s, condition))
                if r is None:
                    continue
                vals.append(getter(r))
            out.append(float(np.mean(vals)) if len(vals) else float("nan"))
        return np.array(out, dtype=float)

    ood_base = per_seed("baseline", lambda r: mean_ood_acc(r["ood_accuracy"]))
    ood_reg = per_seed("he_reg", lambda r: mean_ood_acc(r["ood_accuracy"]))
    mod_base = per_seed("baseline", lambda r: mean_layerwise(r["he_modifier_layerwise"]))
    mod_reg = per_seed("he_reg", lambda r: mean_layerwise(r["he_modifier_layerwise"]))
    seq_base = per_seed("baseline", lambda r: mean_layerwise(r["he_sequence_layerwise"]))
    seq_reg = per_seed("he_reg", lambda r: mean_layerwise(r["he_sequence_layerwise"]))

    return seeds, {
        "ood_baseline": ood_base,
        "ood_reg": ood_reg,
        "modhe_baseline": mod_base,
        "modhe_reg": mod_reg,
        "seqhe_baseline": seq_base,
        "seqhe_reg": seq_reg,
    }


# ----------------------------
# Bracket helpers
# ----------------------------
def add_sig_bracket(ax, x1, x2, y, h, text, lw=1.2, fontsize=11):
    """
    Standard significance bracket from x1 to x2 at y with height h.
    """
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=lw)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fontsize)


def add_vertical_bracket(ax, x, y1, y2, w, text, lw=1.2, fontsize=11, text_dx=0.0, text_dy=0.0):
    """
    Vertical bracket at x spanning y1..y2, extending left by w (a "┐" style).

    Easy-to-tune parameters:
      x: bracket x position
      y1,y2: vertical span
      w: horizontal width (to the left)
      text_dx,text_dy: offset for label placement

    Shape:
        (x, y2) ┐
               │
        (x, y1) ┘
    """
    y_low, y_high = (y1, y2) if y1 <= y2 else (y2, y1)
    ax.plot([x, x - w], [y_low, y_low], linewidth=lw)
    ax.plot([x, x], [y_low, y_high], linewidth=lw)
    ax.plot([x, x - w], [y_high, y_high], linewidth=lw)
    ax.text(x - w + text_dx, y_high + text_dy, text, ha="left", va="bottom", fontsize=fontsize)


def add_horizontal_bracket(ax, x1, x2, y, h, text, lw=1.2, fontsize=11):
    """
    Alias for readability in scatter: horizontal bracket for X-axis comparison.
    """
    add_sig_bracket(ax, x1, x2, y, h, text, lw=lw, fontsize=fontsize)


# ----------------------------
# Plot 1: scatter_ood_vs_modHE with bracket-style p-values
# ----------------------------
def plot_scatter_with_seed_averages_and_ttest_brackets(runs: List[Dict[str, Any]], outdir: str) -> None:
    g = group_runs_by_key(runs)
    noise_levels = unique_sorted_int(runs, "num_noise")
    seeds = unique_sorted_int(runs, "seed")

    # Seed-level vectors and p-values (t-test only)
    _, sv = seed_level_vectors(runs)
    p_ood = paired_ttest_p(sv["ood_baseline"], sv["ood_reg"])
    p_mod = paired_ttest_p(sv["modhe_baseline"], sv["modhe_reg"])

    fig = plt.figure(figsize=(5, 3.5))
    ax = fig.add_subplot(1, 1, 1)

    # --- Per-run arrows (transparent) ---
    per_run_alpha = 0.25
    for n in noise_levels:
        for s in seeds:
            base = g.get((n, s, "baseline"))
            reg = g.get((n, s, "he_reg"))
            if base is None or reg is None:
                continue

            x0 = mean_layerwise(base["he_modifier_layerwise"])
            y0 = mean_ood_acc(base["ood_accuracy"])
            x1 = mean_layerwise(reg["he_modifier_layerwise"])
            y1 = mean_ood_acc(reg["ood_accuracy"])
            if np.isnan([x0, y0, x1, y1]).any():
                continue

            ax.plot([x0], [y0], marker="o", linestyle="None", alpha=per_run_alpha, markersize=5)
            ax.plot([x1], [y1], marker="x", linestyle="None", alpha=per_run_alpha, markersize=6)
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", linewidth=1, alpha=per_run_alpha),
            )

    # --- Seed-averaged arrows (bold) ---
    avg_alpha = 0.95
    seed_avg_points = []  # (x_base, y_base, x_reg, y_reg) per seed
    for s in seeds:
        base_pts = []
        reg_pts = []
        for n in noise_levels:
            base = g.get((n, s, "baseline"))
            reg = g.get((n, s, "he_reg"))
            if base is None or reg is None:
                continue
            base_pts.append((mean_layerwise(base["he_modifier_layerwise"]), mean_ood_acc(base["ood_accuracy"])))
            reg_pts.append((mean_layerwise(reg["he_modifier_layerwise"]), mean_ood_acc(reg["ood_accuracy"])))
        if len(base_pts) == 0 or len(reg_pts) == 0:
            continue

        x0 = float(np.mean([p[0] for p in base_pts]))
        y0 = float(np.mean([p[1] for p in base_pts]))
        x1 = float(np.mean([p[0] for p in reg_pts]))
        y1 = float(np.mean([p[1] for p in reg_pts]))
        seed_avg_points.append((x0, y0, x1, y1))

        ax.plot([x0], [y0], marker="o", linestyle="None", alpha=avg_alpha, markersize=8)
        ax.plot([x1], [y1], marker="x", linestyle="None", alpha=avg_alpha, markersize=9)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", linewidth=2.0, alpha=avg_alpha),
        )

    # ax.set_xlabel("Mean Modifier HE (avg across layers)")
    ax.set_xlabel("Mean Modifier HE")
    ax.set_ylabel("Mean OOD token accuracy")
    # ax.set_ylabel("Mean OOD token accuracy (avg n_primitives=5..12)")
    # ax.set_title("Baseline → +HE-reg: per-run (faint) and per-seed average (bold)")
    ax.grid(True, linewidth=0.5, alpha=0.5)

    # Legend proxies
    h_base = ax.plot([], [], marker="o", linestyle="None")[0]
    h_reg = ax.plot([], [], marker="x", linestyle="None")[0]
    ax.legend([h_base, h_reg], ["baseline", "+HE-reg"], loc="best", frameon=True)

    # ----------------------------
    # Bracket positions
    # ----------------------------
    # We draw two brackets:
    #   (A) vertical bracket for OOD p-value (placed at right side)
    #   (B) horizontal bracket for ModHE p-value (placed near bottom)
    #
    # Edit these values to move brackets:
    bracket_cfg = {
        # Vertical bracket (OOD p-value)
        "v_x_frac": 0.70,      # x position as fraction of x-axis range
        "v_y1_frac": 0.55,     # start y as fraction of y-axis range
        "v_y2_frac": 0.80,     # end y as fraction of y-axis range
        "v_w_frac": 0.02,      # bracket width as fraction of x-axis range
        "v_text_dx": 0.0,      # absolute data units in x (added after converting)
        "v_text_dy": 0.0,      # absolute data units in y
        # Horizontal bracket (ModHE p-value)
        "h_y_frac": 0.90,      # y position as fraction of y-axis range
        "h_h_frac": 0.02,      # bracket height as fraction of y-axis range
        "h_x1_frac": 0.30,     # x1 as fraction of x-axis range
        "h_x2_frac": 0.55,     # x2 as fraction of x-axis range
    }

    # Convert fractions into data coordinates
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xr = xmax - xmin
    yr = ymax - ymin

    # Vertical bracket params
    vx = xmin + bracket_cfg["v_x_frac"] * xr
    vy1 = ymin + bracket_cfg["v_y1_frac"] * yr
    vy2 = ymin + bracket_cfg["v_y2_frac"] * yr
    vw = bracket_cfg["v_w_frac"] * xr
    vtext_dx = bracket_cfg["v_text_dx"]
    vtext_dy = bracket_cfg["v_text_dy"]

    add_vertical_bracket(
        ax,
        x=vx,
        y1=vy1,
        y2=vy2,
        w=vw,
        text=f"OOD \n{fmt_p(p_ood)}",
        lw=1.2,
        fontsize=11,
        text_dx=vtext_dx,
        text_dy=vtext_dy,
    )

    # Horizontal bracket params
    hx1 = xmin + bracket_cfg["h_x1_frac"] * xr
    hx2 = xmin + bracket_cfg["h_x2_frac"] * xr
    hy = ymin + bracket_cfg["h_y_frac"] * yr
    hh = bracket_cfg["h_h_frac"] * yr

    add_horizontal_bracket(
        ax,
        x1=hx1,
        x2=hx2,
        y=hy,
        h=hh,
        text=f"ModHE  {fmt_p(p_mod)}",
        lw=1.2,
        fontsize=11,
    )

    save_fig(fig, outdir, "scatter_ood_vs_modHE")
    plt.close(fig)


# ----------------------------
# Bar chart
# ----------------------------
def plot_bar_he_summary_pretty_ttest_only(runs: List[Dict[str, Any]], outdir: str) -> None:
    _, sv = seed_level_vectors(runs)

    base_mod = sv["modhe_baseline"]
    reg_mod = sv["modhe_reg"]
    base_seq = sv["seqhe_baseline"]
    reg_seq = sv["seqhe_reg"]

    # drop NaNs in pairs (keep pairing for p-values)
    def drop_nan_pair(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return a[m], b[m]

    bm, rm = drop_nan_pair(base_mod, reg_mod)
    bs, rs = drop_nan_pair(base_seq, reg_seq)

    p_mod = paired_ttest_p(bm, rm)
    p_seq = paired_ttest_p(bs, rs)

    # For boxplots we can still display all finite values (paired already dropped)
    data = [bm, rm, bs, rs]
    labels = ["ModHE\nbaseline", "ModHE\n+HE-reg", "SeqHE\nbaseline", "SeqHE\n+HE-reg"]
    positions = [1, 2, 4, 5]  # gap between modifier and sequence groups

    fig = plt.figure(figsize=(5, 3.5))
    ax = fig.add_subplot(1, 1, 1)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.65,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(linewidth=1.8),
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    # Keep colors default-ish but still distinguish baseline vs reg using alpha
    # (No explicit color specification to respect your earlier constraint preference.)
    # We'll just set transparency uniformly and let matplotlib cycle handle facecolors
    for b in bp["boxes"]:
        b.set_alpha(0.65)

    # Overlay seed points (jittered) for readability
    rng = np.random.default_rng(0)
    for pos, vals in zip(positions, data):
        vals = np.asarray(vals, dtype=float)
        jitter = rng.normal(loc=0.0, scale=0.05, size=len(vals))
        ax.plot(np.full_like(vals, pos, dtype=float) + jitter, vals, marker="o", linestyle="None", alpha=0.55)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Avg layerwise HE\n(per-seed avg across noise)")
    # ax.set_title("Baseline vs +HE-reg: internal compositional structure (seed distributions)")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.5)

    # Add significance brackets (p-values from paired t-test)
    # Bracket placement (easy to adjust)
    # Compute ymax per group
    ymax_mod = float(np.nanmax(np.concatenate([bm, rm]))) if len(bm) else 0.0
    ymax_seq = float(np.nanmax(np.concatenate([bs, rs]))) if len(bs) else 0.0
    ymax = max(ymax_mod, ymax_seq, 1e-6)

    ypad = 0.10 * ymax
    h = 0.05 * ymax

    # Modifier bracket between positions 1 and 2
    add_sig_bracket(
        ax,
        x1=positions[0],
        x2=positions[1],
        y=ymax_mod + ypad,
        h=h,
        text=f"t-test {fmt_p(p_mod)}",
        lw=1.2,
        fontsize=11,
    )

    # Sequence bracket between positions 4 and 5
    add_sig_bracket(
        ax,
        x1=positions[2],
        x2=positions[3],
        y=ymax_seq + ypad,
        h=h,
        text=f"t-test {fmt_p(p_seq)}",
        lw=1.2,
        fontsize=11,
    )

    ax.set_ylim(0.002, ymax + ypad + 2.5 * h)

    save_fig(fig, outdir, "bar_he_summary")
    plt.close(fig)


# ----------------------------
# OOD accuracy by complexity: horizontal subplots
# ----------------------------
def plot_ood_by_complexity_horizontal(runs: List[Dict[str, Any]], outdir: str) -> None:
    g = group_runs_by_key(runs)
    noise_levels = unique_sorted_int(runs, "num_noise")
    seeds = unique_sorted_int(runs, "seed")
    n_prims = list(range(5, 13))

    ncols = len(noise_levels)
    fig = plt.figure(figsize=(4.2 * ncols, 4.0))

    for j, n in enumerate(noise_levels, start=1):
        ax = fig.add_subplot(1, ncols, j)

        for cond in ["baseline", "he_reg"]:
            ys_all = []
            for s in seeds:
                r = g.get((n, s, cond))
                if r is None:
                    continue
                ys = []
                for k in n_prims:
                    v = r["ood_accuracy"].get(str(k), r["ood_accuracy"].get(k, np.nan))
                    ys.append(float(v))
                ys_all.append(ys)

            if ys_all:
                mean = np.nanmean(np.array(ys_all, dtype=float), axis=0)
                std = np.nanstd(np.array(ys_all, dtype=float), axis=0)
                ax.errorbar(n_prims, mean, yerr=std, marker="o", linewidth=1, label=cond)

        ax.set_title(f"noise={n}")
        ax.set_xlabel("# primitives")
        if j == 1:
            ax.set_ylabel("Token accuracy")
        ax.grid(True, linewidth=0.5, alpha=0.5)
        ax.set_ylim(0.0, 1.0)
        if j == ncols:
            ax.legend(loc="lower left", frameon=True)

    fig.suptitle("OOD accuracy by test complexity (mean±std over seeds)", y=1.02)
    save_fig(fig, outdir, "ood_accuracy_by_complexity")
    plt.close(fig)


# ----------------------------
# Layerwise HE plots
# ----------------------------
def plot_layerwise_he(runs: List[Dict[str, Any]], outdir: str, which: str) -> None:
    assert which in ("modifier", "sequence")
    g = group_runs_by_key(runs)
    noise_levels = unique_sorted_int(runs, "num_noise")
    seeds = unique_sorted_int(runs, "seed")

    nrows = len(noise_levels)
    fig = plt.figure(figsize=(8.5, 2.6 * nrows))

    for i, n in enumerate(noise_levels, start=1):
        ax = fig.add_subplot(nrows, 1, i)

        for cond in ["baseline", "he_reg"]:
            curves = []
            layers_ref = None
            key = "he_modifier_layerwise" if which == "modifier" else "he_sequence_layerwise"

            for s in seeds:
                r = g.get((n, s, cond))
                if r is None:
                    continue
                xs, ys = sorted_layers(r[key])
                if layers_ref is None:
                    layers_ref = xs
                if layers_ref is not None and len(xs) == len(layers_ref) and np.all(xs == layers_ref):
                    curves.append(ys)

            if curves:
                M = np.vstack(curves)
                mean = np.mean(M, axis=0)
                std = np.std(M, axis=0)
                ax.errorbar(layers_ref, mean, yerr=std, marker="o", linewidth=1, label=cond)

        ax.set_title(f"Layerwise {which.capitalize()} HE (num_noise={n})")
        ax.set_xlabel("Layer")
        ax.set_ylabel("MSE (HE)")
        ax.grid(True, linewidth=0.5, alpha=0.5)
        ax.legend(loc="best", frameon=True)

    save_fig(fig, outdir, f"layerwise_he_{which}")
    plt.close(fig)


# ----------------------------
# Training losses
# ----------------------------
def plot_training_losses(runs: List[Dict[str, Any]], outdir: str) -> None:
    g = group_runs_by_key(runs)
    noise_levels = unique_sorted_int(runs, "num_noise")
    seeds = unique_sorted_int(runs, "seed")

    nrows = len(noise_levels)
    fig = plt.figure(figsize=(9.0, 3.0 * nrows))

    for i, n in enumerate(noise_levels, start=1):
        ax = fig.add_subplot(nrows, 1, i)

        for cond in ["baseline", "he_reg"]:
            ce_curves, he_curves, tot_curves = [], [], []
            for s in seeds:
                r = g.get((n, s, cond))
                if r is None:
                    continue
                logs = r["train_logs"]
                ce = np.array(logs["ce_loss"], dtype=float)
                he = np.array(logs["he_loss"], dtype=float)
                tot = np.array(logs["total_loss"], dtype=float)
                ce_curves.append(ce)
                he_curves.append(he)
                tot_curves.append(tot)

            if ce_curves:
                min_len = min(map(len, ce_curves))
                ceM = np.vstack([c[:min_len] for c in ce_curves])
                heM = np.vstack([c[:min_len] for c in he_curves])
                totM = np.vstack([c[:min_len] for c in tot_curves])

                epochs = np.arange(1, min_len + 1)
                ax.plot(epochs, np.mean(ceM, axis=0), marker="o", linewidth=1, label=f"{cond}: CE")
                ax.plot(epochs, np.mean(totM, axis=0), marker="x", linewidth=1, label=f"{cond}: Total")
                if cond == "he_reg":
                    ax.plot(epochs, np.mean(heM, axis=0), marker="^", linewidth=1, label=f"{cond}: HE-loss")

        ax.set_title(f"Training curves (num_noise={n})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linewidth=0.5, alpha=0.5)
        ax.legend(loc="best", frameon=True)

    save_fig(fig, outdir, "training_losses")
    plt.close(fig)


# ----------------------------
# Summary TSV
# ----------------------------
def write_summary_tsv(runs: List[Dict[str, Any]], outdir: str) -> None:
    path = os.path.join(outdir, "summary.tsv")
    with open(path, "w") as f:
        f.write("\t".join(["num_noise", "seed", "condition", "mean_ood_acc", "mean_mod_he", "mean_seq_he"]) + "\n")
        for r in runs:
            cfg = r["config"]
            mean_acc = mean_ood_acc(r["ood_accuracy"])
            mean_mod = mean_layerwise(r["he_modifier_layerwise"])
            mean_seq = mean_layerwise(r["he_sequence_layerwise"])
            f.write(
                "\t".join(
                    [
                        str(cfg["num_noise"]),
                        str(cfg["seed"]),
                        str(cfg["condition"]),
                        f"{mean_acc:.6f}",
                        f"{mean_mod:.6f}",
                        f"{mean_seq:.6f}",
                    ]
                )
                + "\n"
            )


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True, help="Path to results JSON")
    ap.add_argument("--outdir", type=str, default="plots_he_reg", help="Directory to save plots")
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    apply_ijcai_style()

    with open(args.json, "r") as f:
        blob = json.load(f)
    runs = blob.get("runs", [])
    if not runs:
        raise RuntimeError("No runs found in JSON.")

    plot_scatter_with_seed_averages_and_ttest_brackets(runs, args.outdir)
    plot_bar_he_summary_pretty_ttest_only(runs, args.outdir)
    plot_ood_by_complexity_horizontal(runs, args.outdir)
    plot_layerwise_he(runs, args.outdir, which="modifier")
    plot_layerwise_he(runs, args.outdir, which="sequence")
    plot_training_losses(runs, args.outdir)

    write_summary_tsv(runs, args.outdir)

    print(f"Saved plots + summary to: {args.outdir}")


if __name__ == "__main__":
    main()
