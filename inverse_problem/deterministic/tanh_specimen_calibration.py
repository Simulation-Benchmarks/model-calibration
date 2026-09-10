"""
Deterministic calibration: fit a single Young's modulus E to the tanh data.

The counterpart of the probabilistic tanh_specimen_calibration.py, stripped of all
uncertainty. The forward model is the same linear law

    sigma_model,i(E) = E * eps_i

and E is a *point* parameter constrained to a box [E_lo, E_hi] taken from the
config JSON. We pick the E in that box that minimises the misfit between the
model response and the observed stress,

    J(E) = sum_i  w_i * (sigma_obs,i - E * eps_i) ** 2

with weights w_i = 1 / svar_i when "objective.weighted" is true (svar_i is the
known observation variance folded in by load_data, matching the noise model used
by the Bayesian script) and w_i = 1 otherwise. J is quadratic in E, so the
unconstrained minimiser is closed-form,

    E_optimised_unc = sum_i w_i eps_i sigma_obs,i / sum_i w_i eps_i ** 2

and the constrained answer is that value clipped to the box (also verified with
a bounded 1-D optimiser). The fitted modulus is written as JSON
(``{"E_optimised": ...}``) to this stage's output/ directory; there is no chain
and no posterior.

Run with the `embedded_bias_inference` conda env.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(ROOT, "common"))
from tanh_specimen_common import load_data  # noqa: E402

DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "tanh_specimen_calibration_config.json")
DEFAULT_DATA_DIR = os.path.join(ROOT, "data_processing", "output")
DEFAULT_META_DIR = os.path.join(ROOT, "data", "output")  # DGP metadata (single copy)
DEFAULT_RESULT_DIR = os.path.join(SCRIPT_DIR, "output")
RESULT_FILE = "tanh_specimen_calibration.json"


def load_det_config(path):
    """Parse + validate the deterministic-calibration config JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path) as fh:
        cfg = json.load(fh)

    param = cfg.get("parameter")
    if not param or "bounds" not in param:
        raise ValueError(f"{path}: 'parameter' must be an object with 'bounds'")
    lo, hi = param["bounds"]
    if not (lo < hi):
        raise ValueError(f"{path}: parameter.bounds must be [lo, hi] with lo < hi")
    param.setdefault("name", "E")

    cfg.setdefault("objective", {})
    cfg["objective"].setdefault("weighted", True)
    return cfg


def misfit(E, eps, sigma_obs, w):
    """Weighted sum of squared residuals of the linear law sigma = E * eps."""
    r = sigma_obs - E * eps
    return float(np.sum(w * r * r))


def fit_E(eps, sigma_obs, w, bounds):
    """Minimise the misfit over E in `bounds`; return (E_optimised, diagnostics)."""
    lo, hi = bounds

    # J(E) is quadratic in E -> closed-form unconstrained minimiser.
    a = float(np.sum(w * eps * eps))
    b = float(np.sum(w * eps * sigma_obs))
    E_unc = b / a
    E_clip = min(max(E_unc, lo), hi)

    # Independent check with a bounded 1-D optimiser.
    res = minimize_scalar(
        misfit, bounds=(lo, hi), args=(eps, sigma_obs, w), method="bounded"
    )

    diagnostics = {
        "E_unconstrained": E_unc,
        "E_optimizer": float(res.x),
        "bound_active": bool(E_unc <= lo or E_unc >= hi),
        "optimizer_success": bool(res.success),
    }
    return E_clip, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--meta-dir", default=DEFAULT_META_DIR)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    cfg = load_det_config(args.config)
    data = load_data(args.data_dir, args.meta_dir)

    bounds = tuple(cfg["parameter"]["bounds"])
    weighted = cfg["objective"]["weighted"]
    w = 1.0 / data.svar if weighted else np.ones_like(data.svar)

    E_optimised, _ = fit_E(data.eps, data.sigma_obs, w, bounds)

    result = {"E_optimised": E_optimised}

    out_path = os.path.join(args.result_dir, RESULT_FILE)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    for k, v in result.items():
        print(f"{k:>20}: {v}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
