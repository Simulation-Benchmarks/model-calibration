"""
Shared I/O helpers for the tanh-specimen calibration pipeline.

Pipeline (one-way data flow):

    data/tanh_specimen_dgp.py                     -> data/output/tanh_specimen_{cube,cylinder}.csv,
                                                    data/output/tanh_specimen_metadata.json
    data_processing/
        process_observed_data.py                 -> data_processing/output/
                                                    tanh_specimen_{cube,cylinder}_observed.csv
    inverse_problem/{deterministic,probabilistic}/
        tanh_specimen_calibration.py             -> that method's output/ (point-estimate JSON/CSV, or emcee HDFBackend .h5)
    post_processing/{deterministic,probabilistic}/
        tanh_specimen_postprocess.py             -> summaries, metrics, predictive draws, figures

Both the sampling and the post-processing script load the config and the data
through the two helpers below, so the loaders / validation live in one place.
"""

import json
import os
from collections import namedtuple

import numpy as np

DATA_META = "tanh_specimen_metadata.json"
SPECIMENS = ("cube", "cylinder")
OBSERVED_CSV = "tanh_specimen_{}_observed.csv"

# moment keys each prior family expects in the config
PRIOR_MOMENT_KEYS = {
    "normal": ("mean", "std"),
    "halfnormal": ("std",),
    "lognormal": ("mean", "std"),  # mean/std of the *underlying* normal
}

# pooled dataset handed to the likelihood / post-processing
SpecimenData = namedtuple(
    "SpecimenData", ["eps", "sigma_obs", "svar", "specimen"]
)


def load_config(path):
    """Parse + validate the inference config JSON.

    Returns the parsed dict with an extra ``"names"`` key holding the ordered
    list of parameter names (the canonical theta order used everywhere else).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path) as fh:
        cfg = json.load(fh)

    params = cfg.get("parameters")
    if not params:
        raise ValueError(f"{path}: 'parameters' must be a non-empty list")

    names = []
    for i, spec in enumerate(params):
        for key in ("name", "prior", "moments"):
            if key not in spec:
                raise ValueError(f"{path}: parameters[{i}] missing '{key}'")
        family = spec["prior"]
        if family not in PRIOR_MOMENT_KEYS:
            raise ValueError(
                f"{path}: parameters[{i}] unknown prior '{family}' "
                f"(known: {sorted(PRIOR_MOMENT_KEYS)})"
            )
        for mkey in PRIOR_MOMENT_KEYS[family]:
            if mkey not in spec["moments"]:
                raise ValueError(
                    f"{path}: parameters[{i}] ('{spec['name']}') prior "
                    f"'{family}' needs moments{PRIOR_MOMENT_KEYS[family]}, "
                    f"missing '{mkey}'"
                )
        names.append(spec["name"])

    ndim = len(params)

    sampler = cfg.get("sampler", {})
    for key in ("nwalkers", "nsteps", "discard", "seed"):
        if key not in sampler:
            raise ValueError(f"{path}: sampler missing '{key}'")
    if sampler["nwalkers"] % 2 != 0:
        raise ValueError(f"{path}: sampler.nwalkers must be even (emcee requirement)")
    if sampler["nwalkers"] <= 2 * ndim:
        raise ValueError(
            f"{path}: sampler.nwalkers ({sampler['nwalkers']}) must exceed 2*ndim ({2 * ndim})"
        )
    if sampler["discard"] >= sampler["nsteps"]:
        raise ValueError(
            f"{path}: sampler.discard ({sampler['discard']}) must be < nsteps ({sampler['nsteps']})"
        )

    cfg["names"] = names
    return cfg


def load_data(data_dir, meta_dir):
    """Load the pre-processed specimen observations and pool them for calibration.

    Reads the per-specimen ``tanh_specimen_{name}_observed.csv`` files (columns
    ``eps``, ``sigma_obs_MPa``) written by ``process_observed_data.py`` from
    ``data_dir``, and the DGP ``tanh_specimen_metadata.json`` from ``meta_dir``
    (the single copy under ``data/output/`` -- it is not re-emitted downstream).
    Returns a :class:`SpecimenData` of length-N_total arrays:

        eps        delta / L                              (exact, noiseless strain)
        sigma_obs  F_measured_N / A                       (noisy stress, MPa)
        svar       (sigma_N / A) ** 2                     (known obs variance)
        specimen   specimen label per row

    ``svar`` is recomputed here from the metadata constants
    ``measurement_noise.sigma_N_*_N`` and ``geometric_parameters.A_*_mm2`` --
    the known observation noise is never inferred.
    """
    meta_path = os.path.join(meta_dir, DATA_META)
    csv_paths = {
        name: os.path.join(data_dir, OBSERVED_CSV.format(name)) for name in SPECIMENS
    }
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found -- run tanh_specimen_dgp.py first"
        )
    for p in csv_paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found -- run process_observed_data.py first"
            )

    with open(meta_path) as fh:
        meta = json.load(fh)
    geom = meta["geometric_parameters"]
    noise = meta["measurement_noise"]
    area = {"cube": geom["A_cube_mm2"], "cylinder": geom["A_cyl_mm2"]}
    sigma_n = {"cube": noise["sigma_N_cube_N"], "cylinder": noise["sigma_N_cyl_N"]}

    eps, sigma_obs, svar, specimen = [], [], [], []
    for name in SPECIMENS:
        # columns: eps, sigma_obs_MPa
        arr = np.loadtxt(csv_paths[name], delimiter=",", skiprows=1)
        s = sigma_n[name] / area[name]
        eps.append(arr[:, 0])
        sigma_obs.append(arr[:, 1])
        svar.append(np.full(arr.shape[0], s**2))
        specimen.append(np.full(arr.shape[0], name))

    return SpecimenData(
        eps=np.concatenate(eps),
        sigma_obs=np.concatenate(sigma_obs),
        svar=np.concatenate(svar),
        specimen=np.concatenate(specimen),
    )
