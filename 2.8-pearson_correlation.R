rm(list = ls())

# https://kucharssim.github.io/bayesflow-cognitive-modeling-book/bayesian-cognitive-modeling/02-parameter-estimation/05-examples/01-pearson-correlation.html

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
def context():
    return dict(n = np.random.randint(3, 101))

def prior():
    return dict(rho = np.random.uniform(low = -1, high = 1))

def likelihood(n, rho):
    y = np.random.multivariate_normal(mean = [0, 0], cov = [[1, rho], [rho, 1]], size = n)
    r = np.corrcoef(y[:,0], y[:,1])[0,1]
    return dict(r = r)

simulator = bf.make_simulator([context, prior, likelihood])
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
    .constrain('rho', lower = -1, upper = 1)
    .rename('rho', 'inference_variables')
    .concatenate(['n', 'r'], into = 'inference_conditions')
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
                       batch_size = 512L) {
  py_run_string(
    sprintf(
      "
history = workflow.fit_online(
    epochs = %d,
    batch_size = %d
)
",
    epochs,
    batch_size
    ))
}

fit_online(epochs = 20, batch_size = 512)


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

# summary statistics of data set 1: r = -0.8109671, n = 11
# summary statistics of data set 2: r = -0.8109671, n = 33
# summary statistics of data set 3: r = -0.8109671, n = 99

py_run_string(
  "
inference_data = dict(
    n = np.array([[11], [33], [99]]),
    r = np.array([[-0.8109671], [-0.8109671], [-0.8109671]]))
samples = workflow.sample(num_samples = 2000, conditions = inference_data)
"
)

py$inference_data
py$samples

# Build a tidy posterior table in python and bring back to R 
py_run_string("
import numpy as np, pandas as pd

def samples_to_tidy(samples, inference_data):
    # Heuristic: find the array with the posterior draws
    key = None
    for k in ['inference_variables', 'rho', 'theta', 'posterior_samples', 'parameters']:
        if isinstance(samples, dict) and (k in samples):
            key = k; break
    if key is None:
        # fall back to the first ndarray in the dict
        for k, v in samples.items():
            if isinstance(v, np.ndarray):
                key = k; break
    arr = np.asarray(samples[key])  # shape: [C, S, D] or [S, D] or [C, S]

    # Normalize to [C, S, D]
    if arr.ndim == 1:
        arr = arr[None, :, None]
    elif arr.ndim == 2:
        # Could be [S, D] for a single condition
        if arr.shape[1] == 1:
            arr = arr[None, :, :]
        else:
            arr = arr[None, :, :]  # assume single condition, D>1
    # Now arr is [C, S, D]
    C, S = arr.shape[0], arr.shape[1]

    # Pick the first dimension as rho (this problem is 1D)
    if arr.shape[-1] == 1:
        rho = arr[..., 0]                 # shape [C, S]
    else:
        rho = arr[..., 0]                 # if multi-D, first dim is rho

    # Condition labels from inference_data['n']
    ns = np.asarray(inference_data['n']).reshape(-1)
    if ns.size != C:
        # If needed, broadcast or truncate to match
        if ns.size < C:
            ns = np.pad(ns, (0, C-ns.size), mode='edge')
        else:
            ns = ns[:C]

    df = pd.DataFrame({
        'rho': rho.reshape(-1),
        'n':   np.repeat(ns, S)
    })
    return df

df_samples = samples_to_tidy(samples, inference_data)
")

post <- tibble::as_tibble(py$df_samples)


# Plot histograms 

library("ggplot2")

obs_r <- -0.8109671

p <- ggplot(post, aes(rho)) +
  geom_histogram(bins = 60, linewidth = 0.2, colour = "white", fill = "skyblue") +
  geom_vline(xintercept = obs_r, linetype = "dashed") +
  facet_wrap(~ n, nrow = 1, labeller = label_both) +
  labs(
    title    = "Posterior for correlation \u03C1 by sample size n",
    subtitle = "All conditions used r = -0.8109671 at inference time",
    x = expression(rho),
    y = "Posterior sample count"
  ) +
  theme_classic(base_size = 12)

print(p)
