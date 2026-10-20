"""Compare identical Blackwell nonnegative-LASSO paths under three stops."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "results/srt_testdata_20261020"
OUT = ROOT / "nonnegative_mapping_scale_corrected"
INPUTS = {
    "Old mapping": ROOT / "lasso_paths/gpu_blackwell_csr/path_diagnostics.csv",
    "Corrected mapping": OUT / "path_diagnostics.csv",
    "Dual gap": ROOT / "lasso_paths_dual_gap_1e4/gpu_blackwell_csr/path_diagnostics.csv",
}
COLORS = {"Old mapping": "tab:orange", "Corrected mapping": "tab:blue", "Dual gap": "tab:green"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def values(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows])


def main() -> None:
    data = {label: read_csv(path) for label, path in INPUTS.items()}
    error_rows = read_csv(OUT / "field_error_comparison.csv")
    reference = data["Corrected mapping"]
    x = values(reference, "lambda_ratio")
    lambdas = values(reference, "lambda")
    for label, rows in data.items():
        assert len(rows) == len(reference) == len(error_rows)
        np.testing.assert_allclose(values(rows, "lambda"), lambdas, rtol=1e-12)
    np.testing.assert_allclose(values(error_rows, "lambda"), lambdas, rtol=1e-12)

    errors = {
        "Old mapping": values(error_rows, "old_relative_field_error"),
        "Corrected mapping": values(error_rows, "corrected_relative_field_error"),
        "Dual gap": values(error_rows, "dual_gap_relative_field_error"),
    }
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True, sharex=True)
    noise_norm = values(error_rows, "noise_norm")[0]
    panels = [
        (axes[0, 0], "Relative field error", "relative error", errors),
        (axes[0, 1], "Residual / known noise norm", "ratio", {
            label: values(rows, "residual_norm") / noise_norm
            if label != "Corrected mapping"
            else values(rows, "residual_over_noise")
            for label, rows in data.items()
        }),
    ]
    panels.extend([
        (axes[0, 2], "Returned-point KKT / λ", "ratio", {
            label: values(rows, "kkt_over_lambda") for label, rows in data.items()
        }),
        (axes[1, 0], "Iterations per λ", "iterations", {
            label: values(rows, "iterations") for label, rows in data.items()
        }),
        (axes[1, 1], "Cumulative solve time", "seconds", {
            label: np.cumsum(values(rows, "solve_seconds")) for label, rows in data.items()
        }),
    ])
    for ax, title, ylabel, series in panels:
        for label, yy in series.items():
            ax.plot(x, yy, marker="o", markersize=3.1, linewidth=1.7,
                    color=COLORS[label], label=label)
        ax.set(title=title, ylabel=ylabel, xscale="log")
        ax.grid(True, alpha=0.25)
    axes[0, 1].axhline(1, color="0.4", linestyle=":", linewidth=1, label="Known noise")
    axes[1, 0].axhline(10_000, color="0.4", linestyle=":", linewidth=1)

    ax = axes[1, 2]
    old_difference = values(reference, "relative_difference_from_old")
    dual_difference = values(reference, "relative_difference_from_dual")
    ax.plot(x[1:], 100 * old_difference[1:], color=COLORS["Old mapping"],
            marker="o", markersize=3.1, linewidth=1.7, label="Old vs corrected")
    ax.plot(x[1:], 100 * dual_difference[1:], color=COLORS["Dual gap"],
            marker="o", markersize=3.1, linewidth=1.7, label="Dual gap vs corrected")
    ax.set(title="Coefficient-vector difference", ylabel="relative difference (%)", xscale="log")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    failed = np.array([row["converged"].lower() != "true" for row in reference])
    for ax, _, _, series in panels:
        yy = series["Corrected mapping"]
        ax.scatter(x[failed], yy[failed], marker="x", s=75, linewidths=2,
                   color=COLORS["Corrected mapping"], zorder=5)
    axes[1, 2].scatter(x[failed], 100 * old_difference[failed], marker="x",
                       s=75, linewidths=2, color=COLORS["Corrected mapping"], zorder=5)
    for ax in axes[1]:
        ax.set_xlabel("λ / λmax   (decreasing along path →)")
    # Set the shared axis once; inverting each subplot would toggle it.
    axes[0, 0].set_xlim(x.max() * 1.05, x.min() / 1.05)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.035))
    fig.suptitle("Nonnegative LASSO stopping rules · matched 31-point Blackwell CSR path",
                 fontsize=15)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"stopping_rule_path_comparison.{extension}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
