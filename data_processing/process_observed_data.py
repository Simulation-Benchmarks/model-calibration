"""
Pre-process the raw specimen measurements into calibration-ready observations.

Reads the raw load--displacement CSVs written by data/tanh_specimen_dgp.py
(columns delta_mm, F_measured_N) plus tanh_specimen_metadata.json, and derives
the quantities the calibration model consumes:

    eps           = delta_mm / L          normalized... strain (exact, noiseless)
    sigma_obs_MPa = F_measured_N / A      observed (noisy) stress

One CSV per specimen is written to this stage's output/ directory:

    output/tanh_specimen_cube_observed.csv       eps, sigma_obs_MPa
    output/tanh_specimen_cylinder_observed.csv   (same columns)

The metadata is *not* copied -- downstream `load_data` reads the single copy
under data/output/ directly. No true / noise-free response is written anywhere.

Run with the `embedded_bias_inference` conda env.
"""

import argparse
import json
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_DATA_DIR = os.path.join(ROOT, "data", "output")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

SPECIMENS = ("cube", "cylinder")
RAW_CSV = "tanh_specimen_{}.csv"
OBSERVED_CSV = "tanh_specimen_{}_observed.csv"
META_FILE = "tanh_specimen_metadata.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    meta_path = os.path.join(args.data_dir, META_FILE)
    raw_paths = {
        name: os.path.join(args.data_dir, RAW_CSV.format(name)) for name in SPECIMENS
    }
    for p in (meta_path, *raw_paths.values()):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found -- run tanh_specimen_dgp.py first")

    with open(meta_path) as fh:
        meta = json.load(fh)
    geom = meta["geometric_parameters"]
    L = geom["L_mm"]
    area = {"cube": geom["A_cube_mm2"], "cylinder": geom["A_cyl_mm2"]}

    os.makedirs(args.output_dir, exist_ok=True)
    written = []
    for name in SPECIMENS:
        # raw columns: delta_mm, F_measured_N
        arr = np.loadtxt(raw_paths[name], delimiter=",", skiprows=1)
        delta_mm, f_measured = arr[:, 0], arr[:, 1]
        eps = delta_mm / L
        sigma_obs = f_measured / area[name]

        out_csv = os.path.join(args.output_dir, OBSERVED_CSV.format(name))
        np.savetxt(
            out_csv,
            np.column_stack([eps, sigma_obs]),
            delimiter=",",
            header="eps,sigma_obs_MPa",
            comments="",
        )
        written.append((out_csv, eps.size))

    for out_csv, n in written:
        print(f"Wrote {out_csv}  ({n} rows)")


if __name__ == "__main__":
    main()
