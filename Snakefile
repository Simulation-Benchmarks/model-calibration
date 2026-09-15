# Snakemake workflow for the tanh-specimen calibration pipelines.
#
# Shared steps:
#   generate_data           ->  data/output/{tanh_specimen_cube.csv,
#                                             tanh_specimen_cylinder.csv,
#                                             tanh_specimen_metadata.json}
#   process_observed_data   ->  data_processing/output/{tanh_specimen_cube_observed.csv,
#                                                        tanh_specimen_cylinder_observed.csv}
#
# Deterministic (point-estimate) pipeline:
#   deterministic_fit         -> inverse_problem/deterministic/output/tanh_specimen_calibration.json
#   deterministic_postprocess -> post_processing/deterministic/output/tanh_specimen_calibration.png
#
# Probabilistic (emcee) pipeline:
#   sample_posterior -> inverse_problem/probabilistic/output/tanh_specimen_posterior.h5
#   postprocess      -> post_processing/probabilistic/output/{tanh_specimen_posterior_moments.json,
#                                                              tanh_specimen_{corner,traces,calibration}.png}
#
# `rule all`'s targets are picked by --config inference_mode={deterministic,probabilistic}
# (default: deterministic); both pipelines' rules are always defined, so Snakemake
# only executes the subgraph needed to produce the requested targets.
#
# Run:                     snakemake --cores 1 --config inference_mode=deterministic
#                           snakemake --cores 1 --config inference_mode=probabilistic
# Force a clean rebuild:   snakemake --cores 1 --config inference_mode=probabilistic --forceall
#
# Each rule calls `python`, so activate the `inference_benchmark` conda env
# (see environment.yml) before running snakemake.

DATA_DIR = "data/output"
OBSERVED_DIR = "data_processing/output"

DET_RESULT_DIR = "inverse_problem/deterministic/output"
DET_POST_DIR = "post_processing/deterministic/output"
DET_CONFIG_JSON = "inverse_problem/deterministic/tanh_specimen_calibration_config.json"

PROB_POSTERIOR_DIR = "inverse_problem/probabilistic/output"
PROB_RESULTS_DIR = "post_processing/probabilistic/output"
PROB_CONFIG_JSON = "inverse_problem/probabilistic/tanh_specimen_calibration_config.json"

META_FILE = f"{DATA_DIR}/tanh_specimen_metadata.json"  # single copy; read by load_data
RAW_DATA_FILES = [
    f"{DATA_DIR}/tanh_specimen_cube.csv",
    f"{DATA_DIR}/tanh_specimen_cylinder.csv",
    META_FILE,
]
# pre-processed observations; metadata is NOT re-emitted here
OBSERVED_FILES = [
    f"{OBSERVED_DIR}/tanh_specimen_cube_observed.csv",
    f"{OBSERVED_DIR}/tanh_specimen_cylinder_observed.csv",
]

DET_FIT_FILE = f"{DET_RESULT_DIR}/tanh_specimen_calibration.json"
DET_FIG_FILE = f"{DET_POST_DIR}/tanh_specimen_calibration.png"

PROB_POSTERIOR_FILE = f"{PROB_POSTERIOR_DIR}/tanh_specimen_posterior.h5"
PROB_RESULT_FILES = [
    f"{PROB_RESULTS_DIR}/tanh_specimen_posterior_moments.json",
    f"{PROB_RESULTS_DIR}/tanh_specimen_corner.png",
    f"{PROB_RESULTS_DIR}/tanh_specimen_traces.png",
    f"{PROB_RESULTS_DIR}/tanh_specimen_calibration.png",
]

INFERENCE_MODE = config.get("inference_mode", "deterministic")
if INFERENCE_MODE == "deterministic":
    FINAL_TARGETS = [DET_FIT_FILE, DET_FIG_FILE]
elif INFERENCE_MODE == "probabilistic":
    FINAL_TARGETS = PROB_RESULT_FILES
else:
    raise ValueError(
        f"Unknown inference_mode: {INFERENCE_MODE!r} (expected 'deterministic' or 'probabilistic')"
    )


rule all:
    input:
        FINAL_TARGETS


rule generate_data:
    output:
        RAW_DATA_FILES
    shell:
        "python data/tanh_specimen_dgp.py"


rule process_observed_data:
    input:
        RAW_DATA_FILES
    output:
        OBSERVED_FILES
    shell:
        "python data_processing/process_observed_data.py"


rule deterministic_fit:
    input:
        config=DET_CONFIG_JSON,
        data=OBSERVED_FILES,
        meta=META_FILE,
    output:
        DET_FIT_FILE
    shell:
        "python inverse_problem/deterministic/tanh_specimen_calibration.py"


rule deterministic_postprocess:
    input:
        data=OBSERVED_FILES,
        meta=META_FILE,
        result=DET_FIT_FILE,
    output:
        DET_FIG_FILE
    shell:
        "python post_processing/deterministic/tanh_specimen_postprocess.py"


rule sample_posterior:
    input:
        config=PROB_CONFIG_JSON,
        data=OBSERVED_FILES,
        meta=META_FILE,
    output:
        PROB_POSTERIOR_FILE
    shell:
        "python inverse_problem/probabilistic/tanh_specimen_calibration.py"


rule postprocess:
    input:
        config=PROB_CONFIG_JSON,
        data=OBSERVED_FILES,
        meta=META_FILE,
        posterior=PROB_POSTERIOR_FILE,
    output:
        PROB_RESULT_FILES
    shell:
        "python post_processing/probabilistic/tanh_specimen_postprocess.py"
