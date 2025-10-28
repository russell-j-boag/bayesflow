rm(list = ls())

# https://kucharssim.github.io/bayesflow-cognitive-modeling-book/bayesian-cognitive-modeling/02-parameter-estimation/03-binomials/03-inferring-a-common-rate.html

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
    return dict(n = np.random.randint(1, 101, size = 2))

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

# Here inference is on four datasets (pairs of ks and ns) fed as an array
py_run_string("
inference_data = dict(
    k = np.array([[14, 16], [0, 10], [7, 3], [5, 5]]),
    n = np.array([[20, 20], [10, 10], [10, 10], [10, 10]])
)

samples = workflow.sample(num_samples = 2000, conditions = inference_data)
")

# Bring it into R and display
str(py$samples$theta)
# num [1:4, 1:2000, 1] 0.56 0.591 0.437 0.638 0.629 ...


# Plot --------------------------------------------------------------------

library(ggplot2)
library(dplyr)
library(purrr)
library(ggtext)     # for HTML subscripts in subtitle
library(patchwork)

# --- Extract samples & coerce to [D x S] matrix ---
theta_raw <- py$samples$theta
dims <- dim(theta_raw)

if (length(dims) == 3 && dims[3] == 1L) {
  theta_mat <- theta_raw[,,1, drop = TRUE]   # -> [D x S]
} else if (length(dims) == 2) {
  theta_mat <- theta_raw                      # already [D x S]
} else if (length(dims) == 1) {
  theta_mat <- matrix(theta_raw, nrow = 1L)   # [1 x S]
} else {
  stop("Unexpected theta shape: ", paste(dims, collapse = "x"))
}

D <- nrow(theta_mat)
S <- ncol(theta_mat)

# k/n as before
k_mat <- as.matrix(py$inference_data$k)
n_mat <- as.matrix(py$inference_data$n)
stopifnot(nrow(k_mat) == D, nrow(n_mat) == D)

# Common prior (Beta(1,1))
df_prior <- data.frame(
  theta   = seq(0, 1, length.out = 512),
  density = dbeta(seq(0, 1, length.out = 512), 1, 1)
)

make_panel <- function(d) {
  th_d <- as.numeric(theta_mat[d, ])
  df_post <- tibble(theta = th_d)
  
  kd <- density(th_d, from = 0, to = 1, n = 512)
  df_kde <- tibble(theta = kd$x, density = kd$y)
  
  k1 <- k_mat[d, 1]; k2 <- k_mat[d, 2]
  n1 <- n_mat[d, 1]; n2 <- n_mat[d, 2]
  
  # Conjugate Beta reference (Beta(1 + k_total, 1 + n_total - k_total))
  a_post <- 1 + (k1 + k2)
  b_post <- 1 + ((n1 - k1) + (n2 - k2))
  df_ref <- tibble(theta = df_prior$theta,
                   density = dbeta(df_prior$theta, a_post, b_post))
  
  # Posterior median
  med_theta <- median(th_d, na.rm = TRUE)
  
  # Empirical proportions (ground truth-ish from observed data)
  truth_df <- tibble(
    x    = c(k1 / n1, k2 / n2),   # <- change to c(k1/10, k2/10) if you meant literal “/10”
    grp  = c("k1/n1", "k2/n2")
  )
  
  subtitle_html <- sprintf(
    "k<sub>1</sub>=%d, n<sub>1</sub>=%d; k<sub>2</sub>=%d, n<sub>2</sub>=%d", k1, n1, k2, n2
  )
  
  ggplot(df_post, aes(theta)) +
    geom_histogram(aes(y = after_stat(density)),
                   binwidth = 0.02, boundary = -1, closed = "right",
                   fill = "skyblue", color = "white") +
    # Reference (conjugate) and prior
    geom_line(data = df_ref, aes(theta, density), linewidth = 0.5, linetype = "dashed") +
    geom_line(data = df_prior, aes(theta, density), linewidth = 0.5,
              linetype = "dashed", color = "steelblue") +
    # Posterior median
    geom_vline(xintercept = med_theta, linewidth = 0.6, color = "black") +
    # Empirical proportions (with legend)
    geom_vline(data = truth_df, aes(xintercept = x, color = grp), linewidth = 0.7) +
    scale_color_manual(values = c("k1/n1" = "firebrick", "k2/n2" = "darkgreen"),
                       name = "Empirical prop.") +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
    labs(x = expression(theta), y = "Density",
         title = expression("Posterior of " * theta),
         subtitle = subtitle_html) +
    theme_classic(base_size = 13) +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = ggtext::element_markdown(hjust = 0.5))
}


panels <- purrr::map(1:D, make_panel)
p <- (panels[[1]] | panels[[2]]) / (panels[[3]] | panels[[4]]) +
  plot_layout(guides = "collect") 

print(p)

ggsave(file.path("plots", "posterior.pdf"), plot = p, width = 9, height = 6)
