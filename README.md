# Commands

All scripts run with the `embedded_bias_inference` conda env. Set a shortcut:

```bash
PY=/home/dtyagi/miniconda3/envs/embedded_bias_inference/bin/python
cd /home/dtyagi/inference_benchmark
```

(or `conda activate embedded_bias_inference` and use `python` directly.)

There are two calibration pipelines that share the first two stages:

```text
                                 ┌─ inverse_problem/probabilistic/  ─→ post_processing/probabilistic/
data/  ─→  data_processing/  ─→ ─┤
                                 └─ inverse_problem/deterministic/  ─→ post_processing/deterministic/
```

Every script derives its input/output paths from its own location, so the commands
below work from the repo root **or** from inside a stage directory. Each stage
writes to an `output/` subdir next to its script; `data_processing/` is the shared
pre-processing stage both pipelines read from.

**Model.** The data-generating process is a per-specimen tanh load--displacement
law `F(δ) = A·σ_y·tanh(δ/(L·ε_ref)) + N(0, σ_N²)` (two specimens, cube and
cylinder, differing only in cross-sectional area; noise on the load channel
only). The DGP writes only the raw measured channels (`delta_mm`,
`F_measured_N`); `process_observed_data.py` then derives strain
`ε = δ/L` and observed stress `σ_obs = F_measured/A`. The **probabilistic**
calibration model is an *embedded-model-error* linear law: `σ = E·ε` with
`E ~ LogNormal`, `E[E] = μ_E`, `SD[E] = σ_E`, and we infer `θ = (μ_E, σ_E)`
(physical units, MPa). The likelihood is the independent-normal (moment-matching)
approximation of Sargsyan et al. 2019 (Eqs. 12-13):
`σ_obs,i ~approx N(μ_E·ε_i, σ_E²·ε_i² + s_i²)` with `s_i = σ_N,i/A_i` the *known*
observation-noise SD (recomputed from metadata at load time, not inferred) — the
two moments are exact for the lognormal `E·ε`, so the form matches the normal-`E`
case. The
**deterministic** pipeline fits a single point `E` in a box `[E_lo, E_hi]` by
minimising the (optionally noise-weighted) sum of squared residuals.

---

## 1. Generate the synthetic datasets

```bash
$PY data/tanh_specimen_dgp.py
```

Writes `data/output/tanh_specimen_cube.csv`,
`data/output/tanh_specimen_cylinder.csv` (columns `delta_mm,F_measured_N`) and
`data/output/tanh_specimen_metadata.json`.

Options:

| flag | default | meaning |
|---|---|---|
| `--output-dir DIR` | `data/output` | where to write the datasets |
| `--seed INT` | `0` | RNG seed for the load noise (cube drawn before cylinder) |

Grid sizes / ranges / material constants are module-level constants in the
script, not CLI flags.

---

## 2. Pre-process into calibration observations (shared)

```bash
$PY data_processing/process_observed_data.py
```

Reads the raw CSVs + `data/output/tanh_specimen_metadata.json` and writes, to
`data_processing/output/`: `tanh_specimen_cube_observed.csv`,
`tanh_specimen_cylinder_observed.csv` (columns `eps,sigma_obs_MPa`). The metadata
is **not** copied — downstream `load_data` reads the single copy under
`data/output/` directly (`--meta-dir`). No true / noise-free response is written.
Both the probabilistic and deterministic pipelines read the observation CSVs from
this directory.

Options:

| flag | default | meaning |
|---|---|---|
| `--data-dir DIR` | `data/output` | directory holding the raw specimen CSVs + metadata |
| `--output-dir DIR` | `data_processing/output` | where to write the observation CSVs |

---

## 3a. Probabilistic: sample the posterior (emcee)

```bash
$PY inverse_problem/probabilistic/tanh_specimen_calibration.py
```

Reads the observation CSVs +
`inverse_problem/probabilistic/tanh_specimen_calibration_config.json`, runs emcee,
writes `inverse_problem/probabilistic/output/tanh_specimen_posterior.h5` (emcee
`HDFBackend`; the sampler is reset and re-run on every invocation). Prints a
progress bar.

Options:

| flag | default | meaning |
|---|---|---|
| `--config PATH` | `inverse_problem/probabilistic/tanh_specimen_calibration_config.json` | priors + sampler settings |
| `--data-dir DIR` | `data_processing/output` | directory holding the `*_observed.csv` |
| `--meta-dir DIR` | `data/output` | directory holding `tanh_specimen_metadata.json` |
| `--posterior-dir DIR` | `inverse_problem/probabilistic/output` | where to write the `.h5` |

Priors, walker count, steps and burn-in are edited in the config JSON, not on the
command line. Priors: `μ_E ~ LogNormal` (config `mean`/`std` = moments of
`log μ_E`), `σ_E ~ HalfNormal` (config `std` = scale), both applied as frozen
`scipy.stats` distributions.

---

## 3b. Probabilistic: post-process (summary + figures)

```bash
$PY post_processing/probabilistic/tanh_specimen_postprocess.py
```

Reads the `.h5` + observation CSVs + config and writes to
`post_processing/probabilistic/output/`:

- `tanh_specimen_posterior_moments.json` — posterior mean vector + covariance matrix of `theta = (mu_E, sigma_E)` (keys `names`, `n_samples`, `mean`, `cov`)
- `tanh_specimen_corner.png`, `tanh_specimen_traces.png`, `tanh_specimen_calibration.png`

Options:

| flag | default | meaning |
|---|---|---|
| `--config PATH` | `inverse_problem/probabilistic/tanh_specimen_calibration_config.json` | must match the run that produced the `.h5` |
| `--data-dir DIR` | `data_processing/output` | directory holding the `*_observed.csv` |
| `--meta-dir DIR` | `data/output` | directory holding `tanh_specimen_metadata.json` |
| `--posterior-dir DIR` | `inverse_problem/probabilistic/output` | directory holding `tanh_specimen_posterior.h5` |
| `--output-dir DIR` | `post_processing/probabilistic/output` | where to write the summary + figures |
| `--pp-draws INT` | `6000` | posterior draws used for the calibration-plot predictive band |

---

## 4a. Deterministic: fit a single modulus

```bash
$PY inverse_problem/deterministic/tanh_specimen_calibration.py
```

Reads the observation CSVs +
`inverse_problem/deterministic/tanh_specimen_calibration_config.json`
(`parameter.bounds` = `[E_lo, E_hi]`, `objective.weighted` = bool), minimises the
misfit over `E` in the box, and writes to
`inverse_problem/deterministic/output/`:

- `tanh_specimen_calibration.json` — `E_hat`, bounds, objective value, RMSE, diagnostics
- `tanh_specimen_calibration.csv` — the same scalars as a one-row CSV (for the plot)

Options:

| flag | default | meaning |
|---|---|---|
| `--config PATH` | `inverse_problem/deterministic/tanh_specimen_calibration_config.json` | parameter box + weighting flag |
| `--data-dir DIR` | `data_processing/output` | directory holding the `*_observed.csv` |
| `--meta-dir DIR` | `data/output` | directory holding `tanh_specimen_metadata.json` |
| `--result-dir DIR` | `inverse_problem/deterministic/output` | where to write the JSON + CSV |

---

## 4b. Deterministic: post-process (figure)

```bash
$PY post_processing/deterministic/tanh_specimen_postprocess.py
```

Reads `tanh_specimen_calibration.csv` + the observation CSVs and writes
`post_processing/deterministic/output/tanh_specimen_calibration.png` (measured
points + fitted `σ = E_hat·ε` line). Nothing is re-fitted.

Options:

| flag | default | meaning |
|---|---|---|
| `--data-dir DIR` | `data_processing/output` | directory holding the `*_observed.csv` |
| `--meta-dir DIR` | `data/output` | directory holding `tanh_specimen_metadata.json` |
| `--result-dir DIR` | `inverse_problem/deterministic/output` | directory holding `tanh_specimen_calibration.csv` |
| `--output-dir DIR` | `post_processing/deterministic/output` | where to write the figure |

---

## Full pipelines via Snakemake

The **probabilistic** pipeline is `snakefile_probabilistic.smk` (`generate_data →
process_observed_data → sample_posterior → postprocess`, plus `all`); the
**deterministic** pipeline is `snakefile_deterministic.smk` (`generate_data →
process_observed_data → deterministic_fit → deterministic_postprocess`, plus
`all`). Both share `generate_data` and `process_observed_data`. Run from the repo
root:

```bash
snakemake -s snakefile_probabilistic.smk --cores 1              # build what's missing / stale
snakemake -s snakefile_probabilistic.smk --cores 1 --forceall   # force a clean rebuild
snakemake -s snakefile_probabilistic.smk --cores 1 sample_posterior   # stop after a chosen rule

snakemake -s snakefile_deterministic.smk --cores 1              # deterministic pipeline
snakemake -s snakefile_deterministic.smk --cores 1 --forceall
```

Rules call the `embedded_bias_inference` interpreter by default; override with
`--config python=python`.

## Full pipelines, from scratch (no Snakemake)

```bash
PY=/home/dtyagi/miniconda3/envs/embedded_bias_inference/bin/python
cd /home/dtyagi/inference_benchmark

$PY data/tanh_specimen_dgp.py
$PY data_processing/process_observed_data.py

# probabilistic
$PY inverse_problem/probabilistic/tanh_specimen_calibration.py
$PY post_processing/probabilistic/tanh_specimen_postprocess.py

# deterministic
$PY inverse_problem/deterministic/tanh_specimen_calibration.py
$PY post_processing/deterministic/tanh_specimen_postprocess.py
```

## Inspect the stored posterior

```bash
$PY -c "import emcee; b=emcee.backends.HDFBackend('inverse_problem/probabilistic/output/tanh_specimen_posterior.h5', read_only=True); print(b.get_chain().shape, b.get_log_prob().shape)"
```
