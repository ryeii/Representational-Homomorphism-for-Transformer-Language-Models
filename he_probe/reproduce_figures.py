"""
Reproduce all figures for the Homomorphism Error paper.

This script assumes that the following modules are available in the same
directory and implement the full experimental pipeline:

    - gen_data.py
    - he_metrics.py
    - transformers.py
    - experiment.py

In particular, `experiment.py` must expose:

    run_all_experiments(seed: int, epochs: int = ..., lr: float = ...)

which returns a dict:
{
    "layer":    (layer_accs,    layer_he_mod,    layer_he_seq),
    "sparsity": (sparsity_accs, sparsity_he_mod, sparsity_he_seq),
    "noise":    (noise_accs,    noise_he_mod,    noise_he_seq),
}

Each of the *_accs / *_he_* objects is a nested dictionary of the form
described in the comments in `plot_triplet` and `plot_noise_acc_vs_he`.

Running this file will:
  1. Run all three experiment families for multiple random seeds.
  2. Aggregate the results across seeds.
  3. Save four figures (PDF) in a `figures/` directory:
        - figure3_model_size_ablation.pdf
        - figure4_training_data_sparsity.pdf
        - figure5_noise_injection.pdf
        - figure6_noise_acc_vs_he_poly.pdf
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from he_probe.experiment import run_all_experiments


# ---------------------------------------------------------------------
# Utility: aggregation helpers
# ---------------------------------------------------------------------

AccDict = Dict[int, Dict[int, float]]
HeDict  = Dict[int, Dict[int, float]]


def _stack_over_seeds_dict(
    dicts_per_seed: List[AccDict]
) -> Tuple[List[int], Dict[int, List[Dict[int, float]]]]:
    """
    Take a list of nested dicts (one per seed) and return:
        - sorted list of top-level keys (conditions)
        - mapping condition -> list of inner dicts (one per seed)

    Example (accuracy case):
        dicts_per_seed[s][cond][x] = acc
    """
    if not dicts_per_seed:
        raise ValueError("No dictionaries provided to aggregate.")

    # Assume all seeds share the same set of top-level keys.
    conditions = sorted(dicts_per_seed[0].keys())
    per_cond: Dict[int, List[Dict[int, float]]] = {c: [] for c in conditions}
    for d in dicts_per_seed:
        for c in conditions:
            per_cond[c].append(d[c])
    return conditions, per_cond


def _mean_std_over_seeds_at_x(
    per_seed_dicts: List[Dict[int, float]],
    xs: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given several dicts mapping x -> value (e.g., layer index -> HE),
    compute mean and std over seeds for each x in xs.
    """
    ys = np.array([[d[x] for x in xs] for d in per_seed_dicts], dtype=float)
    return ys.mean(axis=0), ys.std(axis=0)


# ---------------------------------------------------------------------
# Plotting: triplet panels (Accuracy, Modifier HE, Sequence HE)
# ---------------------------------------------------------------------

def plot_triplet(
    acc_dicts: List[AccDict],
    he_mod_dicts: List[HeDict],
    he_seq_dicts: List[HeDict],
    legend_label: str,
    title: str,
    fname: str,
) -> None:
    """
    Make a 1x3 figure:
        [ Accuracy vs test-set primitives,
          Modifier HE vs layer,
          Sequence HE vs layer ]

    Input format (for each experiment family):

        acc_dicts[seed][condition][n_prim_test] = accuracy
        he_mod_dicts[seed][condition][layer_idx] = modifier HE MSE
        he_seq_dicts[seed][condition][layer_idx] = sequence HE MSE
    """
    os.makedirs("figures", exist_ok=True)

    conds, acc_per_cond = _stack_over_seeds_dict(acc_dicts)
    _, he_mod_per_cond = _stack_over_seeds_dict(he_mod_dicts)
    _, he_seq_per_cond = _stack_over_seeds_dict(he_seq_dicts)

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=False)
    ax_acc, ax_mod, ax_seq = axes

    cmap = plt.cm.viridis
    colors = {c: cmap(i / max(1, len(conds) - 1)) for i, c in enumerate(conds)}

    # ---------------- Accuracy panel ----------------
    for c in conds:
        # Use x-values from first seed
        x_vals = sorted(acc_per_cond[c][0].keys())
        mean_y, std_y = _mean_std_over_seeds_at_x(acc_per_cond[c], x_vals)
        ax_acc.errorbar(
            x_vals,
            mean_y,
            yerr=std_y,
            marker="o",
            linewidth=1.5,
            capsize=3,
            elinewidth=1.0,
            label=str(c),
            color=colors[c],
        )
    ax_acc.set_xlabel("Test set primitives")
    ax_acc.set_ylabel("OOD Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.grid(True)

    # ---------------- Modifier HE panel ----------------
    for c in conds:
        x_vals = sorted(he_mod_per_cond[c][0].keys())
        mean_y, std_y = _mean_std_over_seeds_at_x(he_mod_per_cond[c], x_vals)
        ax_mod.errorbar(
            x_vals,
            mean_y,
            yerr=std_y,
            marker="o",
            linewidth=1.5,
            capsize=3,
            elinewidth=1.0,
            label=str(c),
            color=colors[c],
        )
    ax_mod.set_xlabel("Layer")
    ax_mod.set_ylabel("MSE")
    ax_mod.set_title("Modifier HE")
    ax_mod.grid(True)

    # ---------------- Sequence HE panel ----------------
    for c in conds:
        x_vals = sorted(he_seq_per_cond[c][0].keys())
        mean_y, std_y = _mean_std_over_seeds_at_x(he_seq_per_cond[c], x_vals)
        ax_seq.errorbar(
            x_vals,
            mean_y,
            yerr=std_y,
            marker="o",
            linewidth=1.5,
            capsize=3,
            elinewidth=1.0,
            label=str(c),
            color=colors[c],
        )
    ax_seq.set_xlabel("Layer")
    ax_seq.set_ylabel("MSE")
    ax_seq.set_title("Sequence HE")
    ax_seq.grid(True)

    # Shared legend underneath
    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        bbox_to_anchor=(0.5, -0.05),
        title=legend_label,
    )

    # Tweak fonts
    font_size = 10
    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=font_size)
        ax.tick_params(axis="both", which="minor", labelsize=font_size)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    out_path = os.path.join("figures", f"{fname}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------
# Plotting: correlation between noise-level accuracy & modifier HE
# ---------------------------------------------------------------------

from typing import List, Dict
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

AccDict = Dict[int, Dict[int, float]]
HeDict  = Dict[int, Dict[int, float]]

def plot_noise_acc_vs_he(
    noise_accs: List[AccDict],
    noise_he_mod: List[HeDict],
    degrees=(1, 2, 3),
    fname: str = "figure6_noise_acc_vs_he_poly",
    title: str = "Noise: Accuracy vs Modifier HE",
) -> None:
    """
    Figure 6-style plot with the original aggregation & regression logic:

        - For each noise level k and each seed:
            * average accuracy over test primitives
            * average modifier HE over layers
        - Then average those per-seed means across seeds to get:
            xs_mean[k] = mean HE, ys_mean[k] = mean accuracy
            xs_std[k], ys_std[k] for error bars
        - Fit polynomial models (degrees in `degrees`) using
          sklearn PolynomialFeatures + LinearRegression and
          report R^2 via sklearn.metrics.r2_score.
    """
    os.makedirs("figures", exist_ok=True)

    # Sort noise levels; handle both int and str keys robustly
    first_keys = list(noise_accs[0].keys())
    try:
        keys_sorted = sorted(first_keys, key=lambda k: int(k))
    except (TypeError, ValueError):
        keys_sorted = sorted(first_keys)

    xs_mean, xs_std = [], []
    ys_mean, ys_std = [], []

    # Aggregate per noise level
    for k in keys_sorted:
        x_vals, y_vals = [], []

        for seed_idx in range(len(noise_accs)):
            # accuracy: average over test primitives
            acc_vals = list(noise_accs[seed_idx][k].values())
            # modifier HE: average over layers
            he_vals = list(noise_he_mod[seed_idx][k].values())

            y_vals.append(float(np.mean(acc_vals)))
            x_vals.append(float(np.mean(he_vals)))

        xs_mean.append(float(np.mean(x_vals)))
        xs_std.append(float(np.std(x_vals)))
        ys_mean.append(float(np.mean(y_vals)))
        ys_std.append(float(np.std(y_vals)))

    xs_mean = np.asarray(xs_mean, dtype=float)
    ys_mean = np.asarray(ys_mean, dtype=float)
    xs_std = np.asarray(xs_std, dtype=float)
    ys_std = np.asarray(ys_std, dtype=float)

    fig, ax = plt.subplots(figsize=(4, 3))

    # Scatter + error bars for aggregated points
    ax.errorbar(
        xs_mean,
        ys_mean,
        xerr=xs_std,
        yerr=ys_std,
        fmt="o",
        capsize=3,
        elinewidth=1.0,
        label="Noise conditions",
        zorder=3,
    )

    # Polynomial fits with sklearn (matches original approach)
    x_line = np.linspace(xs_mean.min(), xs_mean.max(), 200).reshape(-1, 1)
    X_mean = xs_mean.reshape(-1, 1)

    for deg in degrees:
        poly = PolynomialFeatures(degree=deg)
        X_poly = poly.fit_transform(X_mean)

        model = LinearRegression()
        model.fit(X_poly, ys_mean)

        # Predictions for plotting
        y_line = model.predict(poly.transform(x_line))

        # R^2 on the aggregated points
        y_pred = model.predict(X_poly)
        r2 = r2_score(ys_mean, y_pred)

        ax.plot(
            x_line,
            y_line,
            linewidth=1.5,
            label=f"Deg {deg} (R²={r2:.2f})",
        )

    font_size = 11
    ax.set_xlabel("Mean Modifier Homomorphism Error (MSE)", fontsize=font_size)
    ax.set_ylabel("Mean OOD Generalization Accuracy", fontsize=font_size)
    ax.tick_params(axis="both", which="major", labelsize=font_size - 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=font_size - 1)
    ax.set_title(title, fontsize=font_size + 1)

    out_path = os.path.join("figures", f"{fname}.pdf")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")



# ---------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------

def main():
    # You can increase the number of seeds for smoother error bars.
    # The paper used 5 seeds for the reported results.
    seeds = [0, 1, 2, 3, 4]

    results_per_seed = []
    for s in seeds:
        print("=" * 80)
        print(f"Running experiments for seed {s}")
        res = run_all_experiments(seed=s)
        results_per_seed.append(res)

    # Unpack per-experiment families into lists (one entry per seed)
    layer_accs    = [r["layer"][0]    for r in results_per_seed]
    layer_he_mod  = [r["layer"][1]    for r in results_per_seed]
    layer_he_seq  = [r["layer"][2]    for r in results_per_seed]

    sparsity_accs   = [r["sparsity"][0] for r in results_per_seed]
    sparsity_he_mod = [r["sparsity"][1] for r in results_per_seed]
    sparsity_he_seq = [r["sparsity"][2] for r in results_per_seed]

    noise_accs   = [r["noise"][0] for r in results_per_seed]
    noise_he_mod = [r["noise"][1] for r in results_per_seed]
    noise_he_seq = [r["noise"][2] for r in results_per_seed]

    # ---------------- Figure 3: model size ablation triplet ----------------
    plot_triplet(
        acc_dicts=layer_accs,
        he_mod_dicts=layer_he_mod,
        he_seq_dicts=layer_he_seq,
        legend_label="# of transformer layers",
        title="Model size ablation results",
        fname="figure3_model_size_ablation",
    )

    # ---------------- Figure 4: training data sparsity triplet -------------
    plot_triplet(
        acc_dicts=sparsity_accs,
        he_mod_dicts=sparsity_he_mod,
        he_seq_dicts=sparsity_he_seq,
        legend_label="# of training set primitives",
        title="Training data sparsity results",
        fname="figure4_training_data_sparsity",
    )

    # ---------------- Figure 5: noise injection triplet --------------------
    plot_triplet(
        acc_dicts=noise_accs,
        he_mod_dicts=noise_he_mod,
        he_seq_dicts=noise_he_seq,
        legend_label="# of randomly inserted noise tokens in training set",
        title="Noise injection results",
        fname="figure5_noise_injection",
    )

    # ---------------- Figure 6: accuracy vs modifier HE --------------------
    plot_noise_acc_vs_he(
        noise_acc_dicts=noise_accs,
        noise_he_mod_dicts=noise_he_mod,
        degrees=(1, 2, 3),
        fname="figure6_noise_acc_vs_he_poly",
        title="Noise: Accuracy vs Modifier HE",
    )


if __name__ == "__main__":
    main()
