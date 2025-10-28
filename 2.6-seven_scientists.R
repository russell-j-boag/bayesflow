rm(list = ls())

# https://kucharssim.github.io/bayesflow-cognitive-modeling-book/bayesian-cognitive-modeling/02-parameter-estimation/04-gaussians/02-the-seven-scientists.html

options(reticulate.use_uv = FALSE)
Sys.setenv(RETICULATE_PYTHON = "/Users/rjb779/Library/r-miniconda-arm64/envs/r-bf/bin/python")
## or, if you made the clean env:
# Sys.setenv(RETICULATE_PYTHON="/Users/rjb779/Library/r-miniconda-arm64/envs/r-bf-jax/bin/python")
Sys.setenv(KERAS_BACKEND = "jax")  # belt-and-braces
library(reticulate)
py_config()                      # expect NumPy 1.26.x
keras <- import("keras")
keras$config$backend()           # "jax"

# Import python libraries
py_run_string("
import os
# Silence absl/JAX INFO logs
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # 0=all, 3=errors only (TF-style; harmless with JAX)
os.environ.setdefault('JAX_PLATFORMS', 'cpu')       # newer env name
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')   # legacy env still honored
os.environ.setdefault('KERAS_BACKEND', 'jax')       # ensure Keras uses JAX

import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('jax').setLevel(logging.ERROR)
logging.getLogger('bayesflow').setLevel(logging.WARNING)

import numpy as np
import bayesflow as bf

# Robust pyplot import (headless-safe)
import matplotlib
try:
    import matplotlib.pyplot as plt
except Exception:
    matplotlib.use('Agg'); import matplotlib.pyplot as plt

import keras, jax
print('Backend   :', keras.config.backend())
print('Platform  :', jax.default_backend())   # 'cpu'
print('NumPy     :', np.__version__)
print('BayesFlow :', getattr(bf, '__version__', 'unknown'))
")


# Simulator ---------------------------------------------------------------

make_simulator <- function() {
  reticulate::py_run_string("
def prior():
    mu = np.random.normal(scale = 10)
    sigma = np.random.gamma(shape = 1.5, scale = 5, size = 7)

    return dict(mu = mu, sigma = sigma)

def likelihood(mu, sigma):
    x = np.random.normal(loc = mu, scale = sigma)
    
    return dict(x = x)

simulator = bf.make_simulator([prior, likelihood])
")
}

make_simulator()


# Approximator ------------------------------------------------------------

make_approximator <- function(network = c("CouplingFlow")) {
  network <- match.arg(network)
  cmd <- sprintf(
    "
adapter = (
    bf.Adapter()
    .constrain(['sigma'], lower = 0)
    .standardize(include = ['mu', 'sigma'], mean = 0.0, std = 1.0)
    .concatenate(['mu', 'sigma'], into = 'inference_variables')
    .rename('x', 'inference_conditions')
    )

workflow = bf.BasicWorkflow(
    simulator = simulator,
    adapter = adapter,
    inference_network = bf.networks.%s()
)
", as.character(network))

py_run_string(cmd)
}

make_approximator(network = "CouplingFlow")


# Training ----------------------------------------------------------------

fit_online <- function(epochs = 50L,
                       num_batches_per_epoch = 200L,
                       batch_size = 512L) {
  py_run_string(
    sprintf(
      "
history = workflow.fit_online(
    epochs = %d,
    num_batches_per_epoch = %d,
    batch_size = %d
)
",
    epochs,
    num_batches_per_epoch,
    batch_size
    ))
}

fit_online(epochs = 20, num_batches_per_epoch = 200L, batch_size = 512)


# Validation --------------------------------------------------------------

run_validation <- function(n_test = 1000L,
                           n_samples = 500L,
                           out_dir = "plots",
                           prefix = "bf_diag") {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  py_run_string(
    sprintf(
      "
import os, matplotlib
matplotlib.use('Agg')  # headless-safe
import matplotlib.pyplot as plt
from collections.abc import Mapping

# Ensure output dir
os.makedirs('%s', exist_ok = True)

# Generate test data and diagnostic plots
test_data = simulator.sample(%d)
figs = workflow.plot_default_diagnostics(test_data = test_data, num_samples = %d)

def save_all_figs(figs, outdir = '%s', prefix = '%s'):
    paths = []
    if isinstance(figs, Mapping):          # dict of named figs
        for k, f in figs.items():
            p = os.path.join(outdir, f'{prefix}_{k}.png')
            f.savefig(p, dpi = 200, bbox_inches = 'tight'); paths.append(p)
    elif isinstance(figs, (list, tuple)):  # list/tuple of figs
        for i, f in enumerate(figs, 1):
            p = os.path.join(outdir, f'{prefix}_{i:02d}.png')
            f.savefig(p, dpi = 200, bbox_inches = 'tight'); paths.append(p)
    else:                                  # single Figure
        p = os.path.join(outdir, f'{prefix}.png')
        figs.savefig(p, dpi = 200, bbox_inches = 'tight'); paths.append(p)
    print('Saved:', paths)
    return paths

saved_paths = save_all_figs(figs)
",
      out_dir,
      n_test,
      n_samples,
      out_dir,
      prefix
    )
  )
  
  # Return list of saved file paths to R
  py_to_r(py$saved_paths)
}

paths <- run_validation()
paths


# Inference ---------------------------------------------------------------

py_run_string(
  "
x = np.array([[-27.020, 3.570, 8.191, 9.898, 9.603, 9.945, 10.056]])
inference_data = dict(x = x)
samples = workflow.sample(num_samples = 2000, conditions = inference_data, split = True)
"
)

py$inference_data
py$samples

# Build pandas data frame
py_run_string("
df_desc = workflow.samples_to_data_frame(samples).describe().transpose()
")

# Bring it to R and display
df_desc <- reticulate::py_to_r(py$df_desc)
print(df_desc)
