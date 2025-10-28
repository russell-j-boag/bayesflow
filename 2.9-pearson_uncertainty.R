rm(list = ls())

# https://kucharssim.github.io/bayesflow-cognitive-modeling-book/bayesian-cognitive-modeling/02-parameter-estimation/05-examples/02-pearson-correlation-uncertainty.html

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
    return dict(
        n = np.random.randint(5, 20),
        reliability = np.random.beta(a=5, b=2, size=2)
    )

def prior():
    return dict(rho = np.random.uniform(low=-1, high=1))

def likelihood(n, reliability, rho):
    sd_true = np.sqrt(reliability)
    sd_err = 1 - np.sqrt(reliability)

    y = np.random.multivariate_normal(
        mean=[0, 0], 
        cov=[
            [np.square(sd_true[0]), rho * np.prod(sd_true)], 
            [rho * np.prod(sd_true), np.square(sd_true[1])]], 
        size = n)
    e = np.random.normal(loc=0, scale=sd_err, size=(n, 2))
    x = y+e
    r = np.corrcoef(x[:,0], x[:,1])[0,1]

    return dict(r=r)

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
    .concatenate(['n', 'reliability', 'r'], into = 'inference_conditions')
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

fit_online(epochs = 20, batch_size = 256)


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

py_run_string("
rt = [0.8, 1.0, 0.5, 0.9, 0.7, 0.4, 1.2, 1.4, 0.6, 1.1, 1.3]
iq = [102, 98, 100, 105, 103, 110, 99, 87, 113, 89, 93]

r = np.corrcoef(rt, iq)[1,0]

error_rt = 0.03
error_iq = 1

reliability = [
    1 - np.square(error_rt) / np.var(rt),
    1 - np.square(error_iq) / np.square(15)
]
reliability

inference_data = dict(
    r = np.array([[r]]),
    n = np.array([[len(rt)]]),
    reliability = np.array(reliability)[np.newaxis]
)
")

py$inference_data

sample_posterior <- function(n_samples = 2000L) {
  # 1) Draw samples in Python
  py_run_string(
    sprintf(
      "
samples = workflow.sample(
    num_samples = %d,
    conditions = inference_data
)
",
    as.integer(n_samples)
    ))

# 2) Tidy to a flat table in Python (robust to different BayesFlow dict keys)
py_run_string("
import numpy as np, pandas as pd
def _samples_to_tidy(samples, inference_data):
    key = None
    for k in ['inference_variables', 'rho', 'theta', 'posterior_samples', 'parameters']:
        if isinstance(samples, dict) and (k in samples):
            key = k; break
    if key is None:
        for k, v in samples.items():
            if isinstance(v, np.ndarray):
                key = k; break
    arr = np.asarray(samples[key])  # shapes: [C,S,D] | [S,D] | [S]

    # Normalize to [C,S,D]
    if arr.ndim == 1:
        arr = arr[None, :, None]
    elif arr.ndim == 2:
        arr = arr[None, :, :]

    C, S = arr.shape[0], arr.shape[1]
    D = 1 if arr.ndim < 3 else arr.shape[2]
    rho = arr[..., 0] if D >= 1 else arr.reshape(C, S)

    ns = np.asarray(inference_data['n']).reshape(-1)
    if ns.size != C:
        ns = ns[:C] if ns.size >= C else np.pad(ns, (0, C-ns.size), mode='edge')

    # Add sample index for convenience
    sample_id = np.tile(np.arange(S), C)

    df = pd.DataFrame({
        'rho': rho.reshape(-1),
        'n':   np.repeat(ns, S),
        'sample_id': sample_id
    })
    return df

df_samples = _samples_to_tidy(samples, inference_data)
")

# 3) Bring back to R as a tibble
tibble::as_tibble(py$df_samples)
}


post <- sample_posterior(n_samples = 2000L)


# Plot --------------------------------------------------------------------

plot_posterior_hist <- function(post, obs_r = -0.8109671, bins = 60) {
  stopifnot(all(c("rho", "n") %in% names(post)))
  post <- post |> dplyr::mutate(n = as.integer(n))  # ensure numeric for label_both
  
  ggplot(post, aes(rho)) +
    geom_histogram(bins = bins, linewidth = 0.2, colour = "white", fill = "skyblue") +
    geom_vline(xintercept = obs_r, linetype = "dashed") +
    facet_wrap(~ n, nrow = 1, labeller = label_both) +
    labs(
      title    = "Posterior for correlation \u03C1 by sample size n",
      subtitle = sprintf("All conditions used r = %.7f at inference time", obs_r),
      x = expression(rho),
      y = "Posterior sample count"
    ) +
    theme_classic(base_size = 12)
}

# Usage (assuming `post` exists with columns rho, n):
p <- plot_posterior_hist(post, obs_r = py$inference_data$r, bins = 60)
print(p)


# Export to ONNX ----------------------------------------------------------

# Make sure we’re using the same Python env
Sys.setenv(RETICULATE_PYTHON = "/Users/rjb779/Library/r-miniconda-arm64/envs/r-bf/bin/python")

# Install TensorFlow (macOS ARM build) + tf2onnx + onnxruntime
py_run_string("
import sys, subprocess
pkgs = [
  'tensorflow-macos==2.16.*',   # TF package for Apple Silicon
  'tf2onnx>=1.16.0',
  'onnx>=1.16.0',
  'onnxruntime>=1.18.0'
]
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade'] + pkgs)
print('Installed: tensorflow-macos, tf2onnx, onnx, onnxruntime')
")

# After you've run fit_online(...) and have `workflow` in Python:
onnx_path <- "exports/bayesflow_couplingflow.onnx"
dir.create("exports", showWarnings = FALSE, recursive = TRUE)

py_run_string("
import bayesflow as bf, keras, inspect

exportable = []
errors = []
for name in dir(bf.networks):
    if name.startswith('_'): 
        continue
    try:
        ctor = getattr(bf.networks, name)
        if not inspect.isclass(ctor):
            continue
        # try to instantiate with no args
        obj = ctor()
        ok = isinstance(obj, keras.Model) or hasattr(obj, 'export') or hasattr(obj, 'save')
        exportable.append((name, type(obj).__name__, ok, isinstance(obj, keras.Model)))
    except Exception as e:
        errors.append((name, str(e)))

print('Exportable-ish candidates:')
for row in exportable:
    print(' ', row)
print('\\nErrors trying to instantiate:')
for row in errors[:10]:
    print(' ', row)
")

Sys.setenv(KERAS_BACKEND = "tensorflow")
py_run_string("
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
print('Keras backend:', keras.config.backend())
")

