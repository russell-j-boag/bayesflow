rm(list = ls())

# https://kucharssim.github.io/bayesflow-cognitive-modeling-book/bayesian-cognitive-modeling/02-parameter-estimation/03-binomials/01-inferring-a-rate.html

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
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # 0 = all, 3 = errors only (TF-style; harmless with JAX)
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

# We will amortize over different sample sizes, so we will also randomly 
# draw "n" during simulations.

py_run_string("
def context():
  return dict(n = np.random.randint(1, 101))

def prior():
  return dict(theta = np.random.beta(a = 1, b = 1))

def likelihood(n, theta):
  return dict(k = np.random.binomial(n = n, p = theta))

simulator = bf.make_simulator([context, prior, likelihood])
")


# Approximator ------------------------------------------------------------

# Adapter
py_run_string("
adapter = (
  bf.Adapter()
  .constrain('theta', lower = 0, upper = 1)
  .rename('theta', 'inference_variables')
  .concatenate(['k', 'n'], into = 'inference_conditions')
)
")

# Workflow
py_run_string("
workflow = bf.BasicWorkflow(
  simulator = simulator,
  adapter = adapter,
  inference_network = bf.networks.CouplingFlow()
)
")


# Training ----------------------------------------------------------------

py_run_string("
history = workflow.fit_online(epochs = 20, batch_size = 512)
")


# Validation --------------------------------------------------------------

py_run_string("
test_data = simulator.sample(1000)
figs = workflow.plot_default_diagnostics(test_data = test_data, num_samples = 500)
")

# Default diagnostic plots
py_run_string("
import os, matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving
import matplotlib.pyplot as plt

# Create the output directory if needed
os.makedirs('plots', exist_ok = True)

# 1) Generate test data and diagnostic plots
test_data = simulator.sample(1000)
figs = workflow.plot_default_diagnostics(test_data = test_data, num_samples = 500)

# 2) Save all figures
from collections.abc import Mapping

def save_all_figs(figs, outdir = 'plots', prefix = 'bf_diag'):
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
")


# Inference ---------------------------------------------------------------

py_run_string("
import numpy as np

inference_data = dict(k = np.array([[5]]), n = np.array([[10]]))

samples = workflow.sample(num_samples = 2000, conditions = inference_data)

df = workflow.samples_to_data_frame(samples)
df_desc = df.describe()

print('Built df_desc with shape:', df_desc.shape)
")

# Bring it into R and display
desc <- py$df_desc                    
desc <- as.data.frame(desc)   
print(desc)

# (Optional) nicer display
# install.packages("knitr"); install.packages("kableExtra")
knitr::kable(desc, digits = 3) |> kableExtra::kable_styling(full_width = FALSE)

# Save
# write.csv(desc, file = "deriv/inference_summary_k5_n10.csv", row.names = TRUE)


# Plot --------------------------------------------------------------------

library("ggplot2")

# Pull samples and data from python
theta <- as.numeric(py$samples$theta)   # posterior draws
df     <- data.frame(theta = theta)

# Extract k and n used in inference
k <- tryCatch(as.integer(py$inference_data$k[1,1]), error = function(e) NA_integer_)
n <- tryCatch(as.integer(py$inference_data$n[1,1]), error = function(e) NA_integer_)

# Prior hyperparameters (uniform Beta(1,1))
a0 <- 1; b0 <- 1

# Posterior hyperparameters
a_post <- if (!is.na(k) && !is.na(n)) a0 + k else NA
b_post <- if (!is.na(k) && !is.na(n)) b0 + (n - k) else NA

# Construct density curves 
xgrid <- seq(0, 1, length.out = 400)
dens_prior <- dbeta(xgrid, a0, b0)
dens_post  <- if (!is.na(a_post)) dbeta(xgrid, a_post, b_post) else rep(NA, length(xgrid))
df_dens <- data.frame(
  theta = xgrid,
  prior = dens_prior,
  posterior = dens_post
)

# Build plot 
p <- ggplot(df, aes(theta)) +
  geom_histogram(aes(y = after_stat(density)),
                 binwidth = 0.05, boundary = 0, closed = "right",
                 fill = "skyblue", color = "white") +
  geom_line(data = df_dens, aes(x = theta, y = prior),
            color = "steelblue", linewidth = 0.5, linetype = "dashed") +
  geom_line(data = df_dens, aes(x = theta, y = posterior),
            color = "black", linewidth = 0.5, linetype = "dashed") +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
  labs(
    x = expression(theta),
    y = "Density",
    title = if (!is.na(a_post))
      sprintf("Prior vs Posterior  (k = %d, n = %d)", k, n)
    else "Prior vs Posterior"
  ) +
  theme_classic(base_size = 13) +
  theme(plot.title.position = "plot")

print(p)

# Save
dir.create("plots", showWarnings = FALSE)
ggsave("plots/post_theta_prior_posterior.png", p, width = 6, height = 4, dpi = 200)
