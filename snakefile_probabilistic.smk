# Snakemake workflow for the tanh-specimen *probabilistic* calibration pipeline.
#
#   generate_data          ->  data/output/{tanh_specimen_cube.csv,
#                                            tanh_specimen_cylinder.csv,
#                                            tanh_specimen_metadata.json}
#   process_observed_data  ->  data_processing/output/{tanh_specimen_cube_observed.csv,
#                                                      tanh_specimen_cylinder_observed.csv}
#   sample_posterior       ->  inverse_problem/probabilistic/output/tanh_specimen_posterior.h5
#   postprocess            ->  post_processing/probabilistic/output/{tanh_specimen_posterior_moments.json,
#                                                                    tanh_specimen_{corner,traces,calibration}.png}
#
# The deterministic (point-estimate) pipeline lives in snakefile_deterministic.smk:
#   snakemake -s snakefile_deterministic.smk --cores 1
#
# Run:   snakemake -s snakefile_probabilistic.smk --cores 1
# Force a clean rebuild:   snakemake -s snakefile_probabilistic.smk --cores 1 --forceall
#
# By default each rule calls the `embedded_bias_inference` conda env interpreter.
# Override with:   snakemake -s snakefile_probabilistic.smk --cores 1 --config python=python

PYTHON = config.get(
    "python",
    "/home/dtyagi/miniconda3/envs/embedded_bias_inference/bin/python",
)

DATA_DIR = "data/output"
OBSERVED_DIR = "data_processing/output"
POSTERIOR_DIR = "inverse_problem/probabilistic/output"
RESULTS_DIR = "post_processing/probabilistic/output"

CONFIG_JSON = "inverse_problem/probabilistic/tanh_specimen_calibration_config.json"

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
POSTERIOR_FILE = f"{POSTERIOR_DIR}/tanh_specimen_posterior.h5"
RESULT_FILES = [
    f"{RESULTS_DIR}/tanh_specimen_posterior_moments.json",
    f"{RESULTS_DIR}/tanh_specimen_corner.png",
    f"{RESULTS_DIR}/tanh_specimen_traces.png",
    f"{RESULTS_DIR}/tanh_specimen_calibration.png",
]


rule all:
    input:
        RESULT_FILES


rule generate_data:
    output:
        RAW_DATA_FILES
    shell:
        "{PYTHON} data/tanh_specimen_dgp.py"


rule process_observed_data:
    input:
        RAW_DATA_FILES
    output:
        OBSERVED_FILES
    shell:
        "{PYTHON} data_processing/process_observed_data.py"


rule sample_posterior:
    input:
        config=CONFIG_JSON,
        data=OBSERVED_FILES,
        meta=META_FILE,
    output:
        POSTERIOR_FILE
    shell:
        "{PYTHON} inverse_problem/probabilistic/tanh_specimen_calibration.py"


rule postprocess:
    input:
        config=CONFIG_JSON,
        data=OBSERVED_FILES,
        meta=META_FILE,
        posterior=POSTERIOR_FILE,
    output:
        RESULT_FILES
    shell:
        "{PYTHON} post_processing/probabilistic/tanh_specimen_postprocess.py"
