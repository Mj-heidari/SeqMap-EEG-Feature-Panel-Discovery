"""Model explanation for the SeqMap panels.

Three complementary views of what a fitted forest is actually using:

* permutation importance, computed on held-out release participants only, so
  an importance value is a statement about transfer rather than about fit;
* the same importances projected onto the scalp, which makes the spatial
  structure of an effect (or the absence of one) directly visible;
* a feature-family decomposition separating spectral power, spectral-model
  parameters, connectivity and regional strength across the two conditions.

Importance is a property of a fitted model, not evidence of a physiological
mechanism, and a permutation importance near zero across every candidate is
itself a reportable result.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.inspection import permutation_importance

from .selection import fit_panel, predict_panel

REGION_ORDER = ["LF", "RF", "LC", "RC", "LP", "RP"]
REGION_LABEL = {
    "LF": "Left frontal", "RF": "Right frontal",
    "LC": "Left central", "RC": "Right central",
    "LP": "Left posterior", "RP": "Right posterior",
}
CONDITIONS = {"eo": "Eyes open", "ec": "Eyes closed", "delta": "EC $-$ EO"}
FAMILIES = {
    "theta": "Theta power", "alpha": "Alpha power", "beta": "Beta power",
    "exponent": "Aperiodic exponent", "offset": "Aperiodic offset",
    "alpha_cf": "Alpha peak freq.", "alpha_height": "Alpha peak height",
    "connectivity": "Connectivity (dwPLI)", "strength": "Regional strength",
}

PALETTE = {"ink": "#18324a", "accent": "#ec8d26", "muted": "#6b7a8c"}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": PALETTE["ink"], "text.color": PALETTE["ink"],
        "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"],
        "axes.edgecolor": PALETTE["muted"], "figure.dpi": 200,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    })


# --------------------------------------------------------------------------
def parse_feature(name):
    """Split a feature column into (condition, region(s), family)."""
    short = name[3:] if name.startswith("f__") else name
    condition, kind, tail = short.split("_", 2)
    if kind == "sp":
        match = re.fullmatch(r"(\w\w)_(theta|alpha|beta|exponent|offset|alpha_cf|alpha_height)", tail)
        if not match:
            raise ValueError(f"Unrecognised spectral feature: {name}")
        return condition, [match.group(1)], match.group(2)
    if kind == "co":
        a, b, _band = tail.split("_")
        return condition, [a, b], "connectivity"
    if kind == "gr":
        region = tail.split("_")[0]
        return condition, [region], "strength"
    raise ValueError(f"Unrecognised feature: {name}")


def release_importance(table, columns, config, n_repeats=20, seed=2026):
    """Permutation importance on each untouched release, averaged over folds.

    The model is refitted on the other releases for every fold, so no
    participant contributes to both the fit and the importance estimate.
    """
    from .evaluation import release_splits
    records = []
    for held, train, test in release_splits(table):
        tr, te = table.iloc[train].reset_index(drop=True), table.iloc[test]
        model = fit_panel(tr, tr.label.to_numpy(int), columns, config)
        x = te[columns].to_numpy(float)
        x = np.where(np.isfinite(x), x, np.nan)
        result = permutation_importance(
            model, x, te.label.to_numpy(int), n_repeats=n_repeats,
            random_state=seed, scoring="balanced_accuracy", n_jobs=1)
        for column, mean, std in zip(columns, result.importances_mean, result.importances_std):
            records.append({"held_release": held, "feature": column,
                            "importance": float(mean), "importance_sd": float(std)})
    return pd.DataFrame(records)


def family_matrix(importance):
    """Aggregate feature importance into a condition x family x region cube."""
    rows = []
    for r in importance.itertuples():
        condition, regions, family = parse_feature(r.feature)
        for region in regions:
            rows.append({"condition": condition, "family": family, "region": region,
                         "held_release": r.held_release,
                         "importance": r.importance / len(regions)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def plot_family_heatmap(importance, path, title):
    """Condition x feature-family importance, summed over regions."""
    style()
    cube = family_matrix(importance)
    grid = (cube.groupby(["family", "condition"]).importance.mean()
            .unstack("condition").reindex(index=list(FAMILIES), columns=list(CONDITIONS)))
    values = grid.to_numpy() * 100
    limit = max(np.nanmax(np.abs(values)), 1e-6)
    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    im = ax.imshow(values, cmap="RdBu_r", norm=TwoSlopeNorm(0, -limit, limit), aspect="auto")
    ax.set(xticks=range(len(CONDITIONS)), xticklabels=list(CONDITIONS.values()),
           yticks=range(len(FAMILIES)), yticklabels=list(FAMILIES.values()))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:+.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(values[i, j]) > .6 * limit else PALETTE["ink"])
    ax.set_title(title, fontsize=9, color=PALETTE["ink"], pad=8)
    bar = fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    bar.set_label("Mean drop in held-out balanced accuracy (pp)", fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)
    return grid


def plot_scalp_importance(importance, regions_path, path, title):
    """Project regional importance onto the scalp via the analysis montage."""
    style()
    try:
        import mne
    except ImportError:
        return None
    regions = json.loads(Path(regions_path).read_text())
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    positions = montage.get_positions()["ch_pos"]
    cube = family_matrix(importance)
    by_region = cube.groupby(["condition", "region"]).importance.mean().unstack("region")
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
    limit = max(np.nanmax(np.abs(by_region.to_numpy())), 1e-6) * 100
    channels, values_template = [], []
    for region, members in regions.items():
        for channel in members:
            if channel in positions:
                channels.append((region, channel))
    xy = np.array([positions[c][:2] for _, c in channels])
    for ax, (condition, label) in zip(axes, CONDITIONS.items()):
        if condition not in by_region.index:
            ax.axis("off")
            continue
        series = by_region.loc[condition]
        values = np.array([series.get(region, np.nan) for region, _ in channels]) * 100
        im, _ = mne.viz.plot_topomap(
            values, xy, axes=ax, show=False, cmap="RdBu_r",
            vlim=(-limit, limit), contours=0, sensors="k.", extrapolate="local")
        ax.set_title(label, fontsize=8.5, color=PALETTE["ink"])
    bar = fig.colorbar(im, ax=axes, fraction=.035, pad=.03)
    bar.set_label("Importance (pp)", fontsize=7)
    fig.suptitle(title, fontsize=9, color=PALETTE["ink"], y=1.06)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)
    return by_region


def plot_target_comparison(summary, path):
    """Held-out balanced accuracy for the primary target and positive controls."""
    style()
    targets = list(summary)
    releases = sorted({r for t in targets for r in summary[t]})
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    width = .8 / len(targets)
    colors = [PALETTE["ink"], PALETTE["accent"], PALETTE["muted"]]
    for i, (target, color) in enumerate(zip(targets, colors)):
        heights = [summary[target].get(r, np.nan) for r in releases]
        ax.bar(np.arange(len(releases)) + (i - (len(targets) - 1) / 2) * width,
               heights, width=width * .92, label=target, color=color)
    ax.axhline(.5, ls="--", lw=1, color=PALETTE["muted"])
    ax.set(xticks=range(len(releases)), xticklabels=[r.replace("_L100", "") for r in releases],
           ylim=(0, 1), ylabel="Balanced accuracy", xlabel="Held-out release")
    ax.legend(fontsize=7, ncol=len(targets), frameon=False, loc="upper center",
              bbox_to_anchor=(.5, 1.18))
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


def plot_stability(panels, path, title):
    """Which candidate blocks survive into the panel of each outer fold."""
    style()
    releases = [p["held_release"] for p in panels]
    groups = sorted({g for p in panels for g in p["groups"]})
    matrix = np.array([[int(g in p["groups"]) for p in panels] for g in groups])
    fig, ax = plt.subplots(figsize=(3.3, max(2.0, len(groups) * .26)))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1.4, aspect="auto")
    ax.set(xticks=range(len(releases)),
           xticklabels=[r.replace("_L100", "") for r in releases],
           yticks=range(len(groups)), yticklabels=[g.replace("_", " ") for g in groups],
           xlabel="Held-out release")
    for i in range(len(groups)):
        for j in range(len(releases)):
            ax.text(j, i, "●" if matrix[i, j] else "·", ha="center", va="center",
                    fontsize=8, color="white" if matrix[i, j] else PALETTE["muted"])
    counts = matrix.sum(axis=1)
    for i, c in enumerate(counts):
        ax.text(len(releases) - .3, i, f"{c}/{len(releases)}", fontsize=6.5,
                va="center", color=PALETTE["muted"])
    ax.set_title(title, fontsize=9, pad=8)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)
    return dict(zip(groups, counts.tolist()))


def block_importance(table, groups, config, n_repeats=30, seed=2026):
    """Permutation importance of whole candidate blocks on held-out releases.

    Single-feature permutation understates a redundant representation: when
    many correlated coordinates carry the same information, removing one
    changes nothing. Permuting an entire prespecified block jointly measures
    what that block contributes that no other block supplies.
    """
    from sklearn.metrics import balanced_accuracy_score
    from .evaluation import release_splits
    columns = [c for v in groups.values() for c in v]
    records = []
    for held, train, test in release_splits(table):
        tr, te = table.iloc[train].reset_index(drop=True), table.iloc[test]
        model = fit_panel(tr, tr.label.to_numpy(int), columns, config)
        y = te.label.to_numpy(int)
        base = balanced_accuracy_score(y, predict_panel(model, te, columns) >= .5)
        index = {c: i for i, c in enumerate(columns)}
        x = te[columns].to_numpy(float)
        x = np.where(np.isfinite(x), x, np.nan)
        rng = np.random.default_rng(seed)
        for name, members in groups.items():
            positions = [index[c] for c in members]
            scores = []
            for _ in range(n_repeats):
                shuffled = x.copy()
                shuffled[:, positions] = shuffled[rng.permutation(len(shuffled))][:, positions]
                scores.append(balanced_accuracy_score(y, model.predict_proba(shuffled)[:, 1] >= .5))
            records.append({"held_release": held, "block": name,
                            "baseline": float(base),
                            "importance": float(base - np.mean(scores)),
                            "importance_sd": float(np.std(scores, ddof=1))})
    return pd.DataFrame(records)


def plot_block_importance(frames, path, title):
    """Block importance for several targets side by side."""
    style()
    targets = list(frames)
    blocks = sorted(frames[targets[0]].block.unique())
    fig, ax = plt.subplots(figsize=(3.5, max(2.2, len(blocks) * .27)))
    grid = np.array([[frames[t][frames[t].block == b].importance.mean() * 100
                      for t in targets] for b in blocks])
    limit = max(np.nanmax(np.abs(grid)), 1e-6)
    im = ax.imshow(grid, cmap="RdBu_r", norm=TwoSlopeNorm(0, -limit, limit), aspect="auto")
    ax.set(xticks=range(len(targets)), xticklabels=targets,
           yticks=range(len(blocks)), yticklabels=[b.replace("_", " ") for b in blocks])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:+.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(grid[i, j]) > .6 * limit else PALETTE["ink"])
    ax.set_title(title, fontsize=9, pad=8)
    bar = fig.colorbar(im, ax=ax, fraction=.05, pad=.03)
    bar.set_label("Drop in held-out balanced accuracy (pp)", fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)
    return pd.DataFrame(grid, index=blocks, columns=targets)


def block_solo_performance(table, groups, config):
    """Held-out balanced accuracy of each candidate block used on its own.

    Permutation importance answers what a block adds that no other block
    supplies; in a redundant representation that can be near zero even for a
    decodable target. Fitting each block alone answers the complementary
    question of what information the block contains, and the two together
    separate "no signal anywhere" from "signal spread across many blocks".
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from .evaluation import release_splits
    records = []
    for held, train, test in release_splits(table):
        tr, te = table.iloc[train].reset_index(drop=True), table.iloc[test]
        y_tr, y_te = tr.label.to_numpy(int), te.label.to_numpy(int)
        for name, columns in groups.items():
            model = fit_panel(tr, y_tr, columns, config)
            probability = predict_panel(model, te, columns)
            records.append({
                "held_release": held, "block": name, "n_features": len(columns),
                "balanced_accuracy": float(balanced_accuracy_score(y_te, probability >= .5)),
                "roc_auc": float(roc_auc_score(y_te, probability)) if len(set(y_te)) == 2 else np.nan,
            })
    return pd.DataFrame(records)


def plot_solo_heatmap(frames, path, title):
    """Standalone block performance across targets, centred on chance."""
    style()
    targets = list(frames)
    blocks = sorted(frames[targets[0]].block.unique())
    order = sorted(blocks, key=lambda b: -frames[targets[-1]].loc[
        frames[targets[-1]].block == b, "balanced_accuracy"].mean())
    grid = np.array([[frames[t].loc[frames[t].block == b, "balanced_accuracy"].mean()
                      for t in targets] for b in order])
    spread = max(np.nanmax(np.abs(grid - .5)), .02)
    fig, ax = plt.subplots(figsize=(3.5, max(2.3, len(order) * .28)))
    im = ax.imshow(grid, cmap="RdBu_r", norm=TwoSlopeNorm(.5, .5 - spread, .5 + spread), aspect="auto")
    ax.set(xticks=range(len(targets)), xticklabels=targets,
           yticks=range(len(order)), yticklabels=[b.replace("_", " ") for b in order])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(grid[i, j] - .5) > .6 * spread else PALETTE["ink"])
    ax.set_title(title, fontsize=9, pad=8)
    bar = fig.colorbar(im, ax=ax, fraction=.05, pad=.03)
    bar.set_label("Held-out balanced accuracy", fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)
    return pd.DataFrame(grid, index=order, columns=targets)
