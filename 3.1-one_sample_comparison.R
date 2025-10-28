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
max_n=100
def context():
    return dict(n=np.random.randint(10, max_n))

def prior_nuisance():
    sigma = np.random.standard_cauchy()
    sigma = np.abs(sigma)
    return dict(sigma = sigma)

def prior_null():
    return dict(delta = 0)

def prior_alternative():
    return dict(delta = np.random.standard_cauchy())

def likelihood(sigma, delta, n):
    mu = sigma * delta
    x = np.zeros(max_n)
    x[:n] = np.random.normal(loc = mu, scale = sigma, size = n)

    observed = np.zeros(max_n)
    observed[:n] = 1

    return dict(x = x, observed = observed)

simulator_null = bf.make_simulator([context, prior_nuisance, prior_null, likelihood])
simulator_alt = bf.make_simulator([context, prior_nuisance, prior_alternative, likelihood])

simulator = bf.simulators.ModelComparisonSimulator([simulator_null, simulator_alt])
")
}

make_simulator()


# Approximator ------------------------------------------------------------

make_approximator <- function() {
  cmd <- sprintf(
    "
adapter = (
    bf.Adapter()
    .as_set(['x', 'observed'])
    .rename('n', 'classifier_conditions')
    .concatenate(['x', 'observed'], into = 'summary_variables')
    .drop(['delta','sigma'])
    )

inference_network = keras.Sequential([
    keras.layers.Dense(32, activation = 'gelu')
    for _ in range(6)
])

approximator = bf.approximators.ModelComparisonApproximator(
    num_models = 2,
    classifier_network = inference_network, 
    summary_network = bf.networks.DeepSet(
        summary_dim = 4,
        mlp_widths_equivariant = (32, 32),
        mlp_widths_invariant_inner = (32, 32),
        mlp_widths_invariant_outer = (32, 32),
        mlp_widths_invariant_last = (32, 32)
    ),
    adapter = adapter
)
")

py_run_string(cmd)
}

make_approximator()


# Training ----------------------------------------------------------------

# Offline training: explicitly provide train/val dicts produced by simulator.sample()
# Offline training (model comparison) -------------------------------------
# Uses your global `simulator`, `adapter`, and `approximator` objects.
# - epochs:   training epochs
# - batches:  number of batches *per epoch*
# - batch_size: samples per batch (simulated once per epoch block)
# - lr:       initial learning rate for CosineDecay
# - with_validation: if TRUE, creates a small validation dataset
# - val_batches: number of validation batches (each of size batch_size)
# - seed:     optional NumPy seed for reproducibility

fit_offline <- function(epochs = 30L,
                        batches = 20L,
                        batch_size = 512L,
                        lr = 1e-4,
                        with_validation = FALSE,
                        val_batches = 5L,
                        seed = NULL) {
  
  py_run_string(sprintf("
import numpy as np, pandas as pd
from keras.optimizers.schedules import CosineDecay
from keras.optimizers import Adam

# --- Params from R ---
epochs      = int(%d)
batches     = int(%d)
batch_size  = int(%d)
lr0         = float(%g)
with_val    = %s
val_batches = int(%d)
seed_val    = %s

# Optional seeding (no 'is not' with a literal warning)
if seed_val is not None:
    np.random.seed(int(seed_val))

# Optimizer & compile
schedule   = CosineDecay(initial_learning_rate=lr0, decay_steps=epochs*batches)
optimizer  = Adam(schedule)
approximator.compile(optimizer)

# Training dataset (offline)
train_raw = simulator.sample(batches * batch_size)
train_ds  = bf.datasets.OfflineDataset(data=train_raw, batch_size=batch_size, adapter=adapter)

# Validation dataset (optional)
val_ds = None
if with_val:
    val_raw = simulator.sample(val_batches * batch_size)
    val_ds  = bf.datasets.OfflineDataset(data=val_raw, batch_size=batch_size, adapter=adapter)

# Fit
history = approximator.fit(
    dataset=train_ds,
    epochs=epochs,
    num_batches=batches,
    batch_size=batch_size,
    validation_dataset=val_ds
)

hist_df = pd.DataFrame(history.history)
",
                as.integer(epochs),
                as.integer(batches),
                as.integer(batch_size),
                lr,
                if (with_validation) "True" else "False",
                as.integer(val_batches),
                if (is.null(seed)) "None" else as.character(seed)
  ))

tibble::as_tibble(reticulate::py_to_r(py$hist_df))
}


# plain training (no validation), uses your approximator/adapter/simulator
hist <- fit_offline(epochs = 30, batches = 20, batch_size = 512, lr = 1e-4)

# with validation
hist_val <- fit_offline(epochs = 40, batches = 24, batch_size = 256,
                        lr = 3e-4, with_validation = TRUE, val_batches = 6, seed = 123)


# Validation --------------------------------------------------------------




# Inference ---------------------------------------------------------------



