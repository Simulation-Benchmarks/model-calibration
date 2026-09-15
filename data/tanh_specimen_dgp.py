"""
Generate + persist the synthetic specimen datasets for the tanh calibration.

True data-generating process (uniaxial load--displacement, tanh hardening,
noise on the load channel only), per specimen:

    F(delta) = A * sigma_y * tanh( delta / (L * eps_ref) ) + N(0, sigma_N^2)
    sigma_y  = E * eps_ref                       (derived)
    sigma_N  = NOISE_FRAC * A * sigma_y          (1% of the asymptotic load)

Two specimens (cube, cylinder) differ *only* in cross-sectional area. This
script owns the true process and every data setting -- it is the single source
of truth. It writes only the raw measured channels; strain/stress derivation
lives in inverse_problem/process_observed_data.py.

    output/tanh_specimen_cube.csv       delta_mm, F_measured_N
    output/tanh_specimen_cylinder.csv   (same columns)
    output/tanh_specimen_metadata.json  geometry (L, areas) + per-specimen noise SD

Run with the `embedded_bias_inference` conda env.
"""

import argparse
import json
import os

import numpy as np

# -- material (only two of {E, eps_ref, sigma_y} are free; fix E and eps_ref) --
E = 2.0e5                    # Young's modulus                 [MPa = N/mm^2]
EPS_REF = 0.0013             # yield-strain scale              [-]
SIGMA_Y = E * EPS_REF        # yield stress (derived)          [MPa]  -> 260.0

# -- geometry (rounded to 3 dp; the same values are written to the metadata) --
L = 10.0                         # gauge length, both specimens   [mm]
A_CUBE = 100.0             # square 10 mm x 10 mm           [mm^2] -> 100.0
A_CYL = round(np.pi * (10.0 / 2.0) ** 2, 3)  # circle, 10 mm diameter        [mm^2] -> 78.54

# -- displacement grid (one-sided, starts at 0; near-yield range) --------
DELTA_MAX = 0.02            # [mm]  -> eps_max = 0.2 %
N_CUBE = 61
N_CYL = 41

# -- load noise: additive Gaussian, sigma_N = NOISE_FRAC of A*sigma_y ----
NOISE_FRAC = 0.01
SEED = 0

# dict insertion order fixes the RNG draw order (cube first, then cylinder)
SPECIMENS = {
    "cube": dict(A=A_CUBE, N=N_CUBE),
    "cylinder": dict(A=A_CYL, N=N_CYL),
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def normalized_strain(delta):
    """s = delta / (L * eps_ref) -- the dimensionless argument of tanh."""
    return np.asarray(delta, dtype=float) / (L * EPS_REF)


def load_noise_free(delta, A):
    """F(delta) = A * sigma_y * tanh( delta / (L*eps_ref) )."""
    return A * SIGMA_Y * np.tanh(normalized_strain(delta))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    written = []
    for name, sp in SPECIMENS.items():
        delta = np.linspace(0.0, DELTA_MAX, sp["N"])
        A = sp["A"]
        sigma_n = NOISE_FRAC * A * SIGMA_Y
        f_true = load_noise_free(delta, A)
        f_meas = f_true + rng.normal(0.0, sigma_n, size=delta.size)

        csv_path = os.path.join(args.output_dir, f"tanh_specimen_{name}.csv")
        np.savetxt(
            csv_path,
            np.column_stack([delta, f_meas]),
            delimiter=",",
            header="delta_mm,F_measured_N",
            comments="",
        )
        written.append((csv_path, sp["N"]))

    meta = {
        "geometric_parameters": {
            "L_mm": round(L, 3),
            "A_cube_mm2": round(A_CUBE, 3),
            "A_cyl_mm2": round(A_CYL, 3),
        },
        "measurement_noise": {
            "sigma_N_cube_N": round(NOISE_FRAC * A_CUBE * SIGMA_Y, 3),
            "sigma_N_cyl_N": round(NOISE_FRAC * A_CYL * SIGMA_Y, 3),
        },
    }
    meta_path = os.path.join(args.output_dir, "tanh_specimen_metadata.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")

    for csv_path, n in written:
        print(f"Wrote {csv_path}  ({n} rows)")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
