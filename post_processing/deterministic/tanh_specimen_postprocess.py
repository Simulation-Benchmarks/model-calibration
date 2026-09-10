"""
Post-process the deterministic tanh-specimen calibration.

Reads the fitted modulus written by
inverse_problem/deterministic/tanh_specimen_calibration.py
(output/tanh_specimen_calibration.csv) plus the observation CSVs, and draws a
single figure overlaying

    * the measured stress--strain points (per specimen), and
    * the fitted linear model response  sigma = E_hat * eps

    figure
      output/tanh_specimen_calibration.png

Nothing is re-fitted here; E_hat comes straight from the CSV.

Run with the `embedded_bias_inference` conda env.
"""

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(ROOT, "common"))
from tanh_specimen_common import load_data  # noqa: E402

DEFAULT_DATA_DIR = os.path.join(ROOT, "data_processing", "output")
DEFAULT_META_DIR = os.path.join(ROOT, "data", "output")  # DGP metadata (single copy)
DEFAULT_RESULT_DIR = os.path.join(ROOT, "inverse_problem", "deterministic", "output")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
RESULT_CSV = "tanh_specimen_calibration.csv"
FIG_FILE = "tanh_specimen_calibration.png"

N_GRID = 400
BLUE = "#2a78d6"
GRAY = "#7a7a76"
SPECIMEN_COLOR = {"cube": "#2a78d6", "cylinder": "#7a3fb5"}


def load_result(result_dir):
    """Read the one-row deterministic-fit CSV into a dict of parsed scalars."""
    path = os.path.join(result_dir, RESULT_CSV)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run deterministic/tanh_specimen_calibration.py first"
        )
    with open(path, newline="") as fh:
        header, values = list(csv.reader(fh))[:2]

    def parse(v):
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            return {"True": True, "False": False}.get(v, v)

    return {k: parse(v) for k, v in zip(header, values)}


def plot_fit(path, eps, sigma_obs, specimen, eps_grid, sigma_model, E_hat, rmse):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name in ("cube", "cylinder"):
        m = specimen == name
        if m.any():
            ax.scatter(
                eps[m], sigma_obs[m], s=24, alpha=0.6, linewidths=0,
                color=SPECIMEN_COLOR.get(name, GRAY), label=f"{name} data",
            )
    ax.plot(
        eps_grid, sigma_model, color=BLUE, lw=2.5, ls="--",
        label=rf"Fitted model  $\sigma = E\,\varepsilon$  ($E$ = {E_hat:,.0f} MPa)",
    )
    ax.set_xlabel(r"strain  $\varepsilon$")
    ax.set_ylabel(r"stress  $\sigma$  (MPa)")
    ax.set_title(f"Deterministic calibration  (RMSE = {rmse:.2f} MPa)")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--meta-dir", default=DEFAULT_META_DIR)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = load_result(args.result_dir)
    E_hat = result["E_hat"]
    data = load_data(args.data_dir, args.meta_dir)

    eps_grid = np.linspace(0.0, float(data.eps.max()), N_GRID)
    sigma_model = E_hat * eps_grid

    os.makedirs(args.output_dir, exist_ok=True)
    fig_path = os.path.join(args.output_dir, FIG_FILE)
    plot_fit(
        fig_path, data.eps, data.sigma_obs, data.specimen,
        eps_grid, sigma_model, E_hat, result["rmse_MPa"],
    )
    print(f"E_hat = {E_hat:,.3f} MPa  (RMSE = {result['rmse_MPa']:.3f} MPa)")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
