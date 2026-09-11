"""
Post-process the emcee posterior for the tanh-specimen calibration.

Reads the chain written by tanh_specimen_calibration.py
(output/tanh_specimen_posterior.h5) plus the datasets + config, and writes --
nothing is recomputed by re-sampling:

    JSON
      tanh_specimen_posterior_moments.json   posterior mean vector + covariance of theta

    figures
      tanh_specimen_corner.png
      tanh_specimen_traces.png
      tanh_specimen_calibration.png

theta = (mu_E, sigma_E) is sampled in physical units (MPa) as the mean and std of
E ~ LogNormal. Calibration uses the independent-normal approximation
sigma_obs | theta ~approx N(mu_E*eps, sigma_E^2*eps^2 + s^2), s known. The
posterior-predictive band below is the *true* lognormal pushforward of E through
sigma = E*eps plus the known observation noise, so it is right-skewed.

Run with the `embedded_bias_inference` conda env.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import emcee

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(ROOT, "common"))
from tanh_specimen_common import load_config, load_data  # noqa: E402

DEFAULT_CONFIG = os.path.join(
    ROOT, "inverse_problem", "probabilistic", "tanh_specimen_calibration_config.json"
)
DEFAULT_DATA_DIR = os.path.join(ROOT, "data_processing", "output")
DEFAULT_META_DIR = os.path.join(ROOT, "data", "output")  # DGP metadata (single copy)
DEFAULT_POSTERIOR_DIR = os.path.join(ROOT, "inverse_problem", "probabilistic", "output")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
POSTERIOR_FILE = "tanh_specimen_posterior.h5"

N_GRID = 400
PP_SEED = 20260908  # matches the calibration notebook's RNG seed

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#7a7a76"
SPECIMEN_COLOR = {"cube": "#2a78d6", "cylinder": "#7a3fb5"}


# -- loading -------------------------------------------------------------
def load_posterior(posterior_dir, discard):
    path = os.path.join(posterior_dir, POSTERIOR_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run tanh_specimen_calibration.py first"
        )
    backend = emcee.backends.HDFBackend(path, read_only=True)
    flat = backend.get_chain(discard=discard, flat=True)
    chain_post = backend.get_chain(discard=discard)  # (nsteps-discard, nwalkers, ndim)
    return flat, chain_post


# -- posterior predictive ----------------------------------------------
def _lognormal_params(mu, sigma):
    """Underlying-normal (m, v) of a lognormal with mean ``mu``, variance ``sigma**2``."""
    v = np.log1p((sigma / mu) ** 2)
    m = np.log(mu) - 0.5 * v
    return m, v


def posterior_predictive(flat, eps_grid, s2, n_draws, rng):
    """Draw sigma ~ p(sigma | eps_grid, data). Returns (theta_subset, mu, y_pred).

    E ~ LogNormal(mean=mu_E, std=sigma_E); the response sigma = E*eps is the
    lognormal pushforward, plus additive Gaussian observation noise.
    ``s2`` is the (scalar or per-grid) known observation variance.
    """
    idx = rng.integers(len(flat), size=n_draws)
    S = flat[idx]  # (n_draws, 2): mu_E, sigma_E
    mu = S[:, 0][:, None] * eps_grid[None, :]  # E[sigma | eps] = mu_E * eps
    m, v = _lognormal_params(S[:, 0], S[:, 1])
    E_draw = np.exp(rng.normal(m, np.sqrt(v)))[:, None]  # (n_draws, 1)
    sigma_model = E_draw * eps_grid[None, :]
    y_pred = sigma_model + rng.normal(0.0, np.sqrt(s2), size=sigma_model.shape)
    return S, mu, y_pred


# -- outputs -----------------------------------------------------------
def write_posterior_moments(path, names, flat):
    """Posterior mean vector + covariance matrix of theta, from the flat chain."""
    mean = flat.mean(axis=0)
    cov = np.atleast_2d(np.cov(flat, rowvar=False))
    payload = {
        "names": list(names),
        "n_samples": int(flat.shape[0]),
        "mean": mean.tolist(),
        "cov": cov.tolist(),
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def print_summary(flat, names):
    print(f"\nflat sample shape: {flat.shape}")
    print(f"{'param':<10}{'mean':>14}{'sd':>14}{'2.5%':>14}{'50%':>14}{'97.5%':>14}")
    for j, nm in enumerate(names):
        col = flat[:, j]
        q = np.percentile(col, [2.5, 50, 97.5])
        print(f"{nm:<10}{col.mean():>14.3f}{col.std():>14.3f}{q[0]:>14.3f}{q[1]:>14.3f}{q[2]:>14.3f}")


def plot_corner(flat, names, path):
    import corner

    fig = corner.corner(
        flat,
        labels=[f"{nm} [MPa]" for nm in names],
        quantiles=[0.03, 0.5, 0.97],
        show_titles=True,
        title_fmt=".3g",
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_traces(chain_post, names, path):
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 2.2 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    for i, nm in enumerate(names):
        axes[i].plot(chain_post[:, :, i], alpha=0.3, lw=0.5)
        axes[i].set_ylabel(f"{nm} [MPa]")
    axes[-1].set_xlabel("step (post burn-in)")
    fig.suptitle("Walker traces")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_posterior_predictive(
    path, eps, sigma_obs, specimen, eps_grid, pred_mean, lo, hi
):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name in ("cube", "cylinder"):
        m = specimen == name
        if m.any():
            ax.scatter(
                eps[m], sigma_obs[m], s=24, alpha=0.6, linewidths=0,
                color=SPECIMEN_COLOR.get(name, GRAY), label=f"{name} data",
            )
    ax.plot(eps_grid, pred_mean, color=BLUE, lw=2.5, ls="--",
            label=r"Posterior-mean prediction  $\mu_E\,\varepsilon$")
    ax.fill_between(eps_grid, lo, hi, color=BLUE, alpha=0.15,
                    label=r"95% posterior predictive")
    ax.set_xlabel(r"strain  $\varepsilon$")
    ax.set_ylabel(r"stress  $\sigma$  (MPa)")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--meta-dir", default=DEFAULT_META_DIR)
    parser.add_argument("--posterior-dir", default=DEFAULT_POSTERIOR_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pp-draws", type=int, default=6000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    names = cfg["names"]
    s = cfg["sampler"]
    data = load_data(args.data_dir, args.meta_dir)
    eps, sigma_obs, svar = data.eps, data.sigma_obs, data.svar

    flat, chain_post = load_posterior(args.posterior_dir, s["discard"])

    # -- posterior predictive (drives the calibration plot band) --
    eps_grid = np.linspace(0.0, float(eps.max()), N_GRID)
    s_fixed2 = float(np.mean(svar))  # known obs variance (constant here)

    rng = np.random.default_rng(PP_SEED)
    _, mu_grid, y_pred = posterior_predictive(
        flat, eps_grid, s_fixed2, args.pp_draws, rng
    )
    lo, hi = np.percentile(y_pred, [2.5, 97.5], axis=0)
    pred_mean = mu_grid.mean(axis=0)  # E[sigma | eps] over the posterior

    # -- write everything ----------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    def out(name):
        return os.path.join(args.output_dir, name)

    write_posterior_moments(out("tanh_specimen_posterior_moments.json"), names, flat)
    plot_corner(flat, names, out("tanh_specimen_corner.png"))
    plot_traces(chain_post, names, out("tanh_specimen_traces.png"))
    plot_posterior_predictive(
        out("tanh_specimen_calibration.png"),
        eps, sigma_obs, data.specimen, eps_grid, pred_mean, lo, hi,
    )

    print_summary(flat, names)
    print(f"\nWrote posterior moments + 3 figures to {args.output_dir}/")


if __name__ == "__main__":
    main()
