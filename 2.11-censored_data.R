rm(list = ls())

# https://kucharssim.github.io/bayesflow-cognitive-modeling-book/bayesian-cognitive-modeling/02-parameter-estimation/05-examples/05-censored-data.html

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
    return dict(
        theta = np.random.uniform(low = 0.25, high = 1)
    )

def summary(scores):
    attempts = len(scores)
    if attempts == 1:
        return dict(
            attempts = attempts,
            score = scores[0],
            min = -1,
            max = -1
        )
    else:
        return dict(
            attempts = attempts,
            score = scores[-1],
            min = np.min(scores[:-1]),
            max = np.max(scores[:-1])
        )

def likelihood(theta, n_questions = 50, min_correct = 30):
    scores = []
    score = 0
    while score < min_correct:
        score = np.random.binomial(n = n_questions, p = theta)
        scores.append(score)
    scores = np.array(scores)

    return summary(scores)

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
    .constrain('theta', lower = 0.25, upper = 1)
    .rename('theta', 'inference_variables')
    .concatenate(['attempts', 'score', 'min', 'max'], into = 'inference_conditions')
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

# Make training and validation data
py_run_string("
train_data = simulator.sample(1000)
val_data = simulator.sample(100)
")

# Run fit
# Offline training: explicitly provide train/val dicts produced by simulator.sample()
fit_offline <- function(epochs = 50L,
                        batch_size = 512L) {
  py_run_string(
    sprintf(
      "
history = workflow.fit_offline(
  data = train_data,
  epochs = %d,
  batch_size = %d,
  validation_data = val_data
)",
      as.integer(epochs),
      as.integer(batch_size)
    )
  )
}

fit_offline(epochs = 50, batch_size = 100)


# Validation --------------------------------------------------------------

run_validation <- function(n_test = 1000L,
                           n_samples = 500L,
                           out_dir = "plots",
                           prefix = "censored_diag") {
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
inference_data = dict(
  attempts = np.array([[950]]),
  score = np.array([[30]]),
  min = np.array([[15]]),
  max = np.array([[25]])
)
")

py$inference_data

sample_posterior <- function(n_samples = 2000L) {
  # 1) sample in Python using your attempts/score/min/max conditions
  py_run_string(
    sprintf("
samples = workflow.sample(
    num_samples=%d,
    conditions=dict(
        attempts=inference_data.get('attempts'),
        score=inference_data.get('score'),
        min=inference_data.get('min'),
        max=inference_data.get('max'),
    )
)
", as.integer(n_samples)))

# 2) robust tidy in Python -> normalize to [C,S,D], prefer 'theta'
py_run_string("
import numpy as np, pandas as pd

def _extract_array(samples):
    # Prefer the actual parameter name used in this script: 'theta'
    for k in ['theta','inference_variables','parameters','posterior_samples','rho','kappa']:
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

def _build_condition_labels(inf):
    # if user provided names, use them
    if 'cond_names' in inf:
        return np.asarray(inf['cond_names']).reshape(-1)
    # otherwise synthesize from available fields in this script
    keys = ['attempts','score','min','max']
    # infer number of conditions
    C = None
    for k in keys:
        if k in inf:
            C = np.asarray(inf[k]).reshape(-1).shape[0]; break
    if C is None:
        return np.array(['cond_0'])
    lab = []
    for i in range(C):
        parts = []
        for k in keys:
            if k in inf:
                v = np.asarray(inf[k]).reshape(-1)[i]
                try: v = int(v)
                except Exception: v = float(v)
                parts.append(f'{k}={v}')
        lab.append(', '.join(parts) if parts else f'cond_{i}')
    return np.array(lab)

arr = _extract_array(samples)          # [C, S, D]
C, S = arr.shape[0], arr.shape[1]
theta_samps = arr[..., 0]              # first dim is theta

cond_labels = _build_condition_labels(inference_data)

df_samples = pd.DataFrame({
    'condition_id'   : np.repeat(np.arange(C), S),
    'condition_label': np.repeat(cond_labels,   S),
    'sample_id'      : np.tile(np.arange(S),    C),
    'theta'          : theta_samps.reshape(-1),
})
")

# 3) back to R
tibble::as_tibble(py$df_samples)
}

post <- sample_posterior(n_samples = 2000L)


# Plot --------------------------------------------------------------------

library("ggplot2")
library("dplyr")

plot_posterior_hist_theta <- function(post, bins = 60, add_kde = TRUE) {
  # Median θ per condition
  meds <- post %>%
    group_by(condition_id, condition_label) %>%
    summarise(theta_med = median(theta, na.rm = TRUE), .groups = "drop")
  
  p <- ggplot(post, aes(theta)) +
    geom_histogram(
      bins = bins,
      aes(y = after_stat(density)),   # <-- density instead of counts
      linewidth = 0.2,
      colour = "white",
      fill = "skyblue"
    ) +
    geom_vline(
      data = meds,
      aes(xintercept = theta_med),
      linetype = "dashed"
    )
  
  if (isTRUE(add_kde)) {
    p <- p + geom_density(alpha = 0.5)  # overlay KDE (no fill)
  }
  
  p +
    facet_wrap(~ condition_label, scales = "free_y") +
    labs(
      title = "Posterior for θ (censored-data model)",
      subtitle = "Dashed line = posterior median θ",
      x = expression(theta),
      y = "Density"
    ) +
    theme_classic(base_size = 12)
}

# usage
p <- plot_posterior_hist_theta(post, bins = 60, add_kde = TRUE)
print(p)
