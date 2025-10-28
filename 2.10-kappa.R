rm(list = ls())

# https://kucharssim.github.io/bayesflow-cognitive-modeling-book/bayesian-cognitive-modeling/02-parameter-estimation/05-examples/03-kappa-coefficient.html

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
    return dict(n=np.random.randint(low = 150, high = 600))

def prior():
    alpha = np.random.beta(a = 1, b = 1)
    beta  = np.random.beta(a = 1, b = 1)
    gamma = np.random.beta(a = 1, b = 1)

    pi = np.array([
        alpha * beta,
        alpha * (1-beta),
        (1-alpha) * (1-gamma),
        (1-alpha) * gamma
    ])

    epsilon = alpha * beta + (1-alpha) * gamma
    psi = (pi[0] + pi[1]) * (pi[0] + pi[2]) + (pi[1] + pi[3]) * (pi[2] + pi[3])
    
    kappa = (epsilon - psi)/(1-psi)

    return dict(pi = pi, kappa = kappa)

def likelihood(n, pi):
    y = np.random.multinomial(n = n, pvals = pi)
    return dict(y = y)

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
    .constrain('kappa', lower = -1, upper = 1)
    .rename('kappa', 'inference_variables')
    .rename('y', 'inference_conditions')
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

fit_online(epochs = 50, batch_size = 512)


# Validation --------------------------------------------------------------

run_validation <- function(n_test = 1000L,
                           n_samples = 500L,
                           out_dir = "plots",
                           prefix = "kappa_diag") {
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

paths <- run_validation(n_test = 1000L)
paths


# Inference ---------------------------------------------------------------

py_run_string("
import numpy as np

# Each row is [n11, n12, n21, n22]
inference_data = dict(
    y = np.array([
        [14, 4,   5, 210],   # influenza
        [20, 7, 103, 417],   # hearing loss
        [ 0, 0,  13, 157],   # rare disease
    ]),
    cond_names = np.array(['influenza','hearing loss','rare disease'])
)

def kappa_from_counts(y):
    y = np.asarray(y)
    n11, n12, n21, n22 = y
    n = y.sum()
    if n == 0:
        return np.nan
    Po = (n11 + n22) / n
    Pe = ((n11 + n12)*(n11 + n21) + (n21 + n22)*(n12 + n22)) / (n**2)
    if Pe == 1.0:
        return np.nan
    return (Po - Pe) / (1.0 - Pe)

obs_kappa = np.array([kappa_from_counts(row) for row in inference_data['y']])
n_per     = inference_data['y'].sum(axis=1)

# enrich for convenience
inference_data['obs_kappa'] = obs_kappa
inference_data['n']         = n_per
")

py$inference_data

sample_posterior <- function(n_samples = 2000L) {
  # 1) sample in Python
  py_run_string(
    sprintf("
samples = workflow.sample(
    num_samples = %d,
    conditions = dict(y=inference_data['y'])  # adapter expects 'y'
)
", as.integer(n_samples)))

# 2) robust tidy in Python -> always returns [condition, sample, dim]
py_run_string("
import numpy as np, pandas as pd

def _extract_array(samples):
    # prefer explicit 'kappa' if present; fall back to common keys
    for k in ['kappa','inference_variables','parameters','posterior_samples','theta','rho']:
        if isinstance(samples, dict) and (k in samples):
            arr = np.asarray(samples[k])
            break
    else:
        # last resort: first ndarray value
        for v in samples.values():
            if isinstance(v, np.ndarray):
                arr = np.asarray(v); break
    if arr.ndim == 1:    # [S] -> [C,S,1]
        arr = arr[None, :, None]
    elif arr.ndim == 2:  # [S,D] -> [C,S,D]
        arr = arr[None, :, :]
    return arr

arr = _extract_array(samples)          # [C, S, D]
C, S = arr.shape[0], arr.shape[1]
kappa_samps = arr[..., 0]              # first dim is kappa

cond_names = inference_data.get('cond_names', np.arange(C).astype(str))
n_per      = inference_data.get('n', np.sum(inference_data['y'], axis=1))

df_samples = pd.DataFrame({
    'condition' : np.repeat(cond_names, S),
    'n'         : np.repeat(n_per,     S),
    'sample_id' : np.tile(np.arange(S), C),
    'kappa'     : kappa_samps.reshape(-1),
})
")

# 3) back to R
tibble::as_tibble(py$df_samples)
}

post <- sample_posterior(n_samples = 2000L)

# observed kappas per condition (for plotting reference lines)
obs <- {
  y_mat <- reticulate::py_to_r(py$inference_data$y)
  cond  <- if (!is.null(py$inference_data$cond_names)) {
    as.character(reticulate::py_to_r(py$inference_data$cond_names))
  } else {
    as.character(seq_len(nrow(y_mat)))
  }
  tibble::tibble(
    condition = cond,
    n         = as.integer(rowSums(y_mat)),
    obs_kappa = as.numeric(reticulate::py_to_r(py$inference_data$obs_kappa))
  )
}


# Plot --------------------------------------------------------------------

library("ggplot2")
library("dplyr")

plot_posterior_hist_kappa <- function(post, obs, bins = 60) {
  df <- post %>%
    mutate(
      condition = as.character(condition),
      n = as.integer(n)
    ) %>%
    left_join(obs, by = c("condition","n")) %>%
    mutate(facet_label = paste0(condition, " (n = ", n, ")"))
  
  ggplot(df, aes(kappa)) +
    geom_histogram(bins = bins, linewidth = 0.2, colour = "white", fill = "skyblue") +
    geom_vline(aes(xintercept = obs_kappa), linetype = "dashed", na.rm = TRUE) +
    facet_wrap(~ facet_label, scales = "free_y") +
    labs(
      title = "Posterior for Cohen’s κ by condition",
      subtitle = "Dashed line = observed κ computed from the 2×2 counts",
      x = expression(kappa),
      y = "Posterior sample count"
    ) +
    theme_classic(base_size = 12)
}

p <- plot_posterior_hist_kappa(post, obs, bins = 60)
print(p)
