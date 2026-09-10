# Snakemake workflow for the tanh-specimen *deterministic* (point-estimate) pipeline.
#
#   generate_data           ->  data/output/{tanh_specimen_cube.csv,
#                                             tanh_specimen_cylinder.csv,
#                                             tanh_specimen_metadata.json}
#   process_observed_data   ->  data_processing/output/{tanh_specimen_cube_observed.csv,
#                                                       tanh_specimen_cylinder_observed.csv}
#   deterministic_fit       ->  inverse_problem/deterministic/output/{tanh_specimen_calibration.json,
#                                                                     tanh_specimen_calibration.csv}
#   deterministic_postprocess -> post_processing/deterministic/output/tanh_specimen_calibration.png
#
# The probabilistic (emcee) pipeline lives in snakefile_probabilistic.smk:
#   snakemake -s snakefile_probabilistic.smk --cores 1
#
# Run:   snakemake -s snakefile_deterministic.smk --cores 1
# Force a clean rebuild:   snakemake -s snakefile_deterministic.smk --cores 1 --forceall
#
# By default each rule calls the `embedded_bias_inference` conda env interpreter.
# Override with:   snakemake -s snakefile_deterministic.smk --cores 1 --config python=python

PYTHON = config.get(
    "python",
    "/home/dtyagi/miniconda3/envs/embedded_bias_inference/bin/python",
)

DATA_DIR = "data/output"
OBSERVED_DIR = "data_processing/output"
RESULT_DIR = "inverse_problem/deterministic/output"
POST_DIR = "post_processing/deterministic/output"

CONFIG_JSON = "inverse_problem/deterministic/tanh_specimen_calibration_config.json"

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
FIT_FILES = [
    f"{RESULT_DIR}/tanh_specimen_calibration.json",
    f"{RESULT_DIR}/tanh_specimen_calibration.csv",
]
FIG_FILE = f"{POST_DIR}/tanh_specimen_calibration.png"


rule all:
    input:
        FIT_FILES + [FIG_FILE]


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


rule deterministic_fit:
    input:
        config=CONFIG_JSON,
        data=OBSERVED_FILES,
        meta=META_FILE,
    output:
        FIT_FILES
    shell:
        "{PYTHON} inverse_problem/deterministic/tanh_specimen_calibration.py"


rule deterministic_postprocess:
    input:
        data=OBSERVED_FILES,
        meta=META_FILE,
        result=f"{RESULT_DIR}/tanh_specimen_calibration.csv",
    output:
        FIG_FILE
    shell:
        "{PYTHON} post_processing/deterministic/tanh_specimen_postprocess.py"
