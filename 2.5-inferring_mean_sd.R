rm(list = ls())

# https://kucharssim.github.io/bayesflow-cognitive-modeling-book/bayesian-cognitive-modeling/02-parameter-estimation/04-gaussians/01-inferring-mean-and-standard-deviation.html

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
import numpy as np
import bayesflow as bf

def context(batch_size):
    n = np.random.randint(10, 501, size=batch_size)
    return dict(n=n)

def prior():
    mu = np.random.normal()
    sigma = np.random.gamma(shape=2, scale=1)
    return dict(mu=mu, sigma=sigma)

def summary(y):
    mean = np.mean(y); sd = np.std(y)
    return dict(mean=mean, sd=sd)

def likelihood(n, mu, sigma):
    y = np.random.normal(loc=mu, scale=sigma, size=n)
    return summary(y)

simulator = bf.simulators.make_simulator([prior, likelihood], meta_fn=context)
", local = FALSE)
}

make_simulator()


# Approximator ------------------------------------------------------------

make_approximator <- function(network = c("CouplingFlow")) {
  network <- match.arg(network)
  cmd <- sprintf(
    "
adapter=(
    bf.Adapter()
    .broadcast('n', to='mean')
    .constrain(['sigma', 'sd'], lower=0)
    .standardize(include=['mu', 'sigma'], mean=0.0, std=1.0)
    .concatenate(['mu', 'sigma'], into='inference_variables')
    .concatenate(['n', 'mean', 'sd'], into='inference_conditions')
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

fit_online <- function(epochs = 20,
                       batch_size = 512) {
  cmd <- sprintf(
    "
history = workflow.fit_online(epochs = %d, batch_size = %d)
",
    as.integer(epochs),
    as.integer(batch_size)
  )
  
  py_run_string(cmd)
}

fit_online(epochs = 20, batch_size = 512)


# UP TO HERE --------------------------------------------------------------

# Validation --------------------------------------------------------------

run_diagnostics <- function(n_test = 1000L) {

  cmd <- sprintf("
test_data = simulator.sample(%d)

", n_test)
  
  py_run_string(cmd)
}

run_diagnostics()
py$test_data


# Inference ---------------------------------------------------------------

py_run_string("
import numpy as np

def summary(y):
    return {'mean': np.mean(y), 'sd': np.std(y)}

y = np.array([1.1, 1.9, 2.3, 1.8])
df = summary(y)
df['n'] = y.shape[0]

print('y:', y)
print('summary dict:', df)
")

# Step 2 — build inference_data (separate cell)
reticulate::py_run_string("
import numpy as np
inference_data = {k: np.array(df[k]).reshape(1, 1) for k in df.keys()}
print({k: v.shape for k, v in inference_data.items()})
")

# Step 3 — bring it into R
idata <- reticulate::py_to_r(py$inference_data)
str(idata)

reticulate::py_run_string("
import numpy as np

conditions = {
    'n':    inference_data['n'].astype(np.int32),
    'mean': inference_data['mean'].astype(np.float32),
    'sd':   inference_data['sd'].astype(np.float32),
}

samples = workflow.sample(num_samples=2000, conditions=conditions)

# Show per-key shapes from the returned dict
print({k: np.array(v).shape for k, v in samples.items()})
")

# 2) Pull samples into R and summarise quickly
post <- reticulate::py_to_r(py$samples)
lapply(post, function(a) c(mean = mean(a), sd = sd(a)))

# Drop the leading/trailing singleton dims -> numeric vectors of length 2000
mu_samps    <- drop(post$mu)       # from [1,2000,1] -> [2000]
sigma_samps <- drop(post$sigma)

# Quick summaries
sum_tbl <- rbind(
  mu    = c(mean = mean(mu_samps),    sd = sd(mu_samps),
            ql = as.numeric(quantile(mu_samps, 0.025)), qu = as.numeric(quantile(mu_samps, 0.975))),
  sigma = c(mean = mean(sigma_samps), sd = sd(sigma_samps),
            ql = as.numeric(quantile(sigma_samps, 0.025)), qu = as.numeric(quantile(sigma_samps, 0.975)))
)
print(round(sum_tbl, 3))

hist(mu_samps, breaks = 40, main = "Posterior: mu", xlab = "mu")
hist(sigma_samps, breaks = 40, main = "Posterior: sigma", xlab = "sigma")

