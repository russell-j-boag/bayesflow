rm(list = ls())

# https://kucharssim.github.io/bayesflow-cognitive-modeling-book/bayesian-cognitive-modeling/02-parameter-estimation/03-binomials/02-difference-between-rates.html

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

py_run_string("
def context():
    return dict(
        n = np.random.randint(1, 101, size = 2)
    )

def prior():
    theta = np.random.beta(a = 1, b = 1, size = 2)

    return dict(theta = theta, delta = theta[0] - theta[1])

def likelihood(n, theta):
    k = np.random.binomial(n = n, p = theta)

    return dict(k = k)

simulator = bf.make_simulator([context, prior, likelihood])
")


# Approximator ------------------------------------------------------------

# Adapter
py_run_string("
adapter = (
    bf.Adapter()
    .constrain('delta', lower = -1, upper = 1)
    .rename('delta', 'inference_variables')
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

# Generate test data and diagnostic plots
test_data = simulator.sample(1000)
figs = workflow.plot_default_diagnostics(test_data = test_data, num_samples = 500)

# Save all figures
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

inference_data = dict(
    k = np.array([[5, 7]]),
    n = np.array([[10, 10]])
)

samples = workflow.sample(num_samples = 2000, conditions = inference_data)

df = workflow.samples_to_data_frame(samples)
df_desc = df.describe()

print('Built df_desc with shape:', df_desc.shape)
")

# Bring it into R and display
desc <- py$df_desc                    
desc <- as.data.frame(desc)   
print(desc)


# Plot --------------------------------------------------------------------

library("ggplot2")
library("ggtext")

# Pull posterior samples of delta (or compute from theta)
delta <- tryCatch(as.numeric(py$samples$delta), error = function(e) NULL)
if (is.null(delta)) {
  theta_mat <- tryCatch(py$samples$theta, error = function(e) NULL)
  stopifnot(!is.null(theta_mat))
  theta_mat <- as.matrix(theta_mat)
  stopifnot(ncol(theta_mat) >= 2)
  delta <- theta_mat[,1] - theta_mat[,2]
}

df_post <- data.frame(delta = delta)

# Extract (k1,k2) and (n1,n2) from inference_data for titles/overlays
k1 <- tryCatch(as.integer(py$inference_data$k[1,1]), error = function(e) NA_integer_)
k2 <- tryCatch(as.integer(py$inference_data$k[1,2]), error = function(e) NA_integer_)
n1 <- tryCatch(as.integer(py$inference_data$n[1,1]), error = function(e) NA_integer_)
n2 <- tryCatch(as.integer(py$inference_data$n[1,2]), error = function(e) NA_integer_)
delta_true = (k1 - k2)/10

# Triangular prior for Δ when θ1,θ2 ~ Beta(1,1): f(d)=1-|d|, d∈[-1,1]
xgrid <- seq(-1, 1, length.out = 401)
tri_prior <- pmax(0, 1 - abs(xgrid))
df_prior  <- data.frame(delta = xgrid, density = tri_prior)

# Conjugate Monte-Carlo reference for posterior of Δ via Beta posteriors
# θ1 ~ Beta(1+k1, 1+n1-k1), θ2 ~ Beta(1+k2, 1+n2-k2)
ref_df <- NULL
if (all(!is.na(c(k1,k2,n1,n2)))) {
  a1 <- 1 + k1; b1 <- 1 + (n1 - k1)
  a2 <- 1 + k2; b2 <- 1 + (n2 - k2)
  set.seed(1)
  th1 <- rbeta(200000, a1, b1)
  th2 <- rbeta(200000, a2, b2)
  dref <- th1 - th2
  kd   <- density(dref, from = -1, to = 1, n = 1024, adjust = 1)
  ref_df <- data.frame(delta = kd$x, density = kd$y)
}

# Posterior kernel density for Δ
kd_post <- density(df_post$delta, from = -1, to = 1, n = 1024, adjust = 1)
df_kde  <- data.frame(delta = kd_post$x, density = kd_post$y)

# 95% credible interval
ci <- quantile(df_post$delta, c(0.025, 0.975), na.rm = TRUE)

# Posterior median line
med_delta <- median(df_post$delta, na.rm = TRUE)

# Plotmath title with Δ = θ₁ − θ₂, and optional counts
title_expr <- expression("Posterior of " ~ Delta ~ "(" * theta[1] - theta[2] * ")")
subtitle_html <- if (all(!is.na(c(k1,k2,n1,n2)))) {
  sprintf("k<sub>1</sub>=%d, n<sub>1</sub>=%d; k<sub>2</sub>=%d, n<sub>2</sub>=%d",
          k1, n1, k2, n2)
} else NULL

# Build plot
p <- ggplot(df_post, aes(delta)) +
  geom_histogram(
    aes(y = after_stat(density)),
    binwidth = 0.05,
    boundary = -1,
    closed = "right",
    fill = "skyblue",
    color = "white"
  ) +
  # KDE
  # geom_line(
  #   data = df_kde,
  #   aes(delta, density),
  #   linewidth = 0.5,
  #   linetype = "dashed",
  #   color = "black"
  # ) +
  # Prior
  geom_line(
    data = df_prior,
    aes(delta, density),
    linewidth = 0.5,
    linetype = "dashed",
    color = "steelblue"
  ) +
  # Reference lines
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    linewidth = 0.5,
    color = "grey40"
  ) +
  # Credible intervals
  # geom_vline(xintercept = as.numeric(ci), linetype = "dashed", linewidth = 0.5) +
  # Posterior median
  geom_vline(
    xintercept = med_delta,
    linetype = "solid",
    linewidth = 0.5,
    color = "black"
  ) +
  # Ground truth
  geom_vline(
    xintercept = delta_true,
    linetype = "dashed",
    linewidth = 0.5,
    color = "firebrick"
  ) +
  scale_x_continuous(limits = c(-1, 1), breaks = seq(-1, 1, 0.2)) +
  labs(
    x = expression(Delta ~  ~ "(" * theta[1] - theta[2] * ")"),
    y = "Density",
    title = title_expr,
    subtitle = subtitle_html
  ) +
  theme_classic(base_size = 13) +
  theme(
    plot.title.position = "plot",
    plot.subtitle = ggtext::element_markdown(hjust = 0.5)
  )

print(p)

# Save
dir.create("plots", showWarnings = FALSE)
ggsave("plots/post_delta_prior_posterior.png", p, width = 7, height = 4.5, dpi = 220)
