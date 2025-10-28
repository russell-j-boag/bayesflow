
# install.packages("reticulate")
## Turn off reticulate's uv helper (must be BEFORE library(reticulate))
options(reticulate.use_uv = FALSE)
library("reticulate")

# Start a fresh R session

# If no conda is configured, install Miniconda
if (is.na(conda_binary())) {
  install_miniconda()   # downloads & installs Miniconda managed by reticulate
}

# Create a clean env and point reticulate to its python
conda_create("r-bf", python_version = "3.10")

# Path to python
py <- file.path(miniconda_path(),
                "envs",
                "r-bf",
                if (.Platform$OS.type == "windows")
                  "python.exe"
                else
                  "bin/python")



## Point to your conda env's Python
Sys.setenv(RETICULATE_PYTHON = "/Users/rjb779/Library/r-miniconda-arm64/envs/r-bf/bin/python")

## Lock this interpreter
use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)

## Confirm it stuck
py_config()                       # verify it shows r-bf/python

# Install python libraries in the already selected env
py_install(
  c(
    # core
    "bayesflow==2.0.7",
    "tensorflow-macos==2.16.*",
    "tensorflow-metal==1.2.*",
    
    # recs
    "numpy<2",
    "scipy",
    "pandas",
    "matplotlib",
    "seaborn",
    "tqdm",
    "rich"
  ),
  pip = TRUE
)

# Install a TF-compatible NumPy (1.26.x works best with TF 2.16 on macOS)
PY <- "/Users/rjb779/Library/r-miniconda-arm64/envs/r-bf/bin/python"

# Install a TF-compatible NumPy (1.26.x)
system2(PY, c("-m","pip","install","--upgrade","pip"))
system2(PY, c("-m","pip","install","numpy==1.26.4"))

# (Optional) ensure the rest of the BayesFlow stack is present in this env:
py_install(c(
  "bayesflow==2.0.7",
  "tensorflow-macos==2.16.*",
  "tensorflow-metal==1.2.*",
  "scipy", "pandas", "matplotlib", "seaborn", "tqdm", "rich"
), pip = TRUE)


# -------------------------------------------------------------------------

# 1) pick the env’s Python *before* loading reticulate
options(reticulate.use_uv = FALSE)
Sys.setenv(RETICULATE_PYTHON = "/Users/rjb779/Library/r-miniconda-arm64/envs/r-bf/bin/python")

# 2) load and lock
library(reticulate)
use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)

# 3) sanity: can we import numpy *inside* reticulate now?
py_module_available("numpy")            # should be TRUE
py_run_string("import sys, numpy as np; print(sys.executable); print(np.__version__)")

# Verify it works
py_config()

py_run_string("
import bayesflow, tensorflow as tf, numpy as np
print('BayesFlow:', bayesflow.__version__)
print('TensorFlow:', tf.__version__)
print('NumPy:', np.__version__)
")

# optional: instantiate tiny BayesFlow pieces to confirm Keras backend ok
py_run_string('
from bayesflow.networks import CouplingFlow, DeepSet
DeepSet(elementwise_widths=[16,16], pooling=\"mean\", global_widths=[16,16])
CouplingFlow(depth=2, transform=\"affine\", subnet=\"mlp\", subnet_kwargs={\"widths\":[16,16]})
print(\"BayesFlow networks constructed OK\")
')

