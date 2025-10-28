rm(list = ls())

# https://kucharssim.github.io/bayesflow-cognitive-modeling-book/bayesian-cognitive-modeling/02-parameter-estimation/03-binomials/04-prior-and-posterior-prediction.html

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
py_run_string(
  "
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

# Robust pyplot import
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
"
)


# Simulator ---------------------------------------------------------------

make_simulator <- function() {
  cmd <- sprintf(
    "
def prior():
    return dict(theta = np.random.beta(a = 1, b = 1))

def likelihood(theta):
    return dict(k = np.random.binomial(n = 20, p = theta))

simulator = bf.make_simulator([prior, likelihood])
"
  )
  py_run_string(cmd)
}

make_simulator()


# Prior samples -----------------------------------------------------------

sample_prior <- function(n_samples = 5000) {
  cmd <- sprintf(
    "
samples = simulator.sample(%d)
prior = samples['theta']
prior_predictives = samples['k']
",
    as.integer(n_samples)
  )
  
  py_run_string(cmd)
  
  # Convert to data frames
  prior_df <- as.data.frame(py$prior)
  prior_pred_df <- as.data.frame(py$prior_predictives)
  names(prior_df) <- NULL
  names(prior_pred_df) <- NULL
  
  list(prior = prior_df, prior_predictives = prior_pred_df)
}

prior <- sample_prior(n_samples = 10000)  # run with n_samples=10000
head(prior$prior)
head(prior$prior_predictives)


# Approximator ------------------------------------------------------------

make_approximator <- function(network = c("CouplingFlow")) {
  network <- match.arg(network)
  cmd <- sprintf(
    "
adapter = (
    bf.Adapter()
    .constrain('theta', lower = 0, upper = 1)
    .rename('theta', 'inference_variables')
    .rename('k', 'inference_conditions')
)

workflow = bf.BasicWorkflow(
    simulator = simulator,
    adapter = adapter,
    inference_network = bf.networks.%s()
)
", as.character(network))

py_run_string(cmd)
}

make_approximator()


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


# Inference ---------------------------------------------------------------

sample_posterior <- function(num_samples = 5000, k = 1) {
  cmd <- sprintf(
    "
# Conditions (single observation with count k)
inference_data = dict(k = np.array([[%d]]))

# Posterior over theta given k
posterior = workflow.sample(num_samples = %d, conditions = inference_data)
posterior = posterior['theta'][0, :, 0]

# Posterior predictive draws for k given sampled thetas
posterior_predictives = simulator.sample(%d, theta = posterior)['k']
",
    as.integer(k),
    as.integer(num_samples),
    as.integer(num_samples)
  )
  
  py_run_string(cmd)
  
  # Convert to unnamed data frames
  posterior_df <- as.data.frame(py$posterior)
  posterior_pred_df <- as.data.frame(py$posterior_predictives)
  colnames(posterior_df) <- NULL
  colnames(posterior_pred_df) <- NULL
  
  list(posterior = posterior_df, posterior_predictives = posterior_pred_df)
}

posterior <- sample_posterior(num_samples = 5000, k = 1)
head(posterior$posterior)
head(posterior$posterior_predictives)


# Plot --------------------------------------------------------------------

library("dplyr")
library("tibble")
library("purrr")
library("ggplot2")

# Extract vectors from prior/posterior objects 
# Theta draws
prior_vec     <- as.numeric(unlist(prior$prior))
posterior_vec <- as.numeric(unlist(posterior$posterior))

# Predictive k draws
prior_k     <- as.integer(as.numeric(unlist(prior$prior_predictives)))
posterior_k <- as.integer(as.numeric(unlist(posterior$posterior_predictives)))

# Prior/posterior theta histograms
brks_theta <- seq(0, 1.05, by = 0.05)

df_plot <- bind_rows(
  tibble(theta = prior_vec, dist = "Prior"),
  tibble(theta = posterior_vec, dist = "Posterior")
)

p1 <- ggplot(df_plot, aes(theta, fill = dist)) +
  geom_histogram(
    aes(y = after_stat(density)),
    position = "identity",
    alpha = 0.4,
    color = "black",
    breaks = brks_theta
  ) +
  scale_fill_manual(values = c(Prior = "red", Posterior = "blue")) +
  labs(
    title = "Prior and posterior",
    x = expression(paste("Rate ", theta)),
    y = "Density",
    fill = NULL
  ) +
  theme_classic(base_size = 12)

print(p1)

# Prior/posterior predictives (k) histograms
# If you know n_trials from the simulator, set it here; otherwise infer:
n_trials <- max(c(prior_k, posterior_k), na.rm = TRUE)
brks_k <- seq(-0.5, n_trials + 0.5, by = 1)

df_plot_k <- bind_rows(
  tibble(k = prior_k, dist = "Prior predictives"),
  tibble(k = posterior_k, dist = "Posterior predictives")
)

p2 <- ggplot(df_plot_k, aes(k, fill = dist)) +
  geom_histogram(
    aes(y = after_stat(density)),      # with binwidth = 1, density = probability
    position = "identity",
    alpha = 0.4,
    color = "black",
    breaks = brks_k
  ) +
  scale_x_continuous(breaks = 0:n_trials, limits = c(-0.5, n_trials + 0.5)) +
  scale_fill_manual(values = c("Prior predictives" = "red", "Posterior predictives" = "blue")) +
  labs(
    title = "Prior and posterior predictives",
    x = "k (successes)",
    y = "Probability",
    fill = NULL
  ) +
  theme_classic(base_size = 12)

print(p2)


# Parameter recovery ------------------------------------------------------

# Posterior draws for a given observed k, using your sample_posterior() wrapper
infer_theta_from_k <- function(k_obs, num_samples) {
  res <- sample_posterior(num_samples = num_samples, k = k_obs)
  draws <- as.numeric(unlist(res$posterior))
  # Safety: clamp to (0,1) in case amortizer emitted unconstrained values
  pmin(pmax(draws, 0), 1)
}

# Run recovery across a grid of true thetas and return summary
recover_theta_grid <- function(theta_grid = seq(0.05, 0.95, by = 0.05),
                               n_trials = 20,
                               num_post = 5000,
                               seed = 123) {
  set.seed(seed)
  map_dfr(theta_grid, function(th) {
    k_obs <- rbinom(1, size = n_trials, prob = th)
    draws <- infer_theta_from_k(k_obs, num_samples = num_post)
    
    q025 <- quantile(draws, 0.025)
    q975 <- quantile(draws, 0.975)
    cover_lbl <- ifelse(th >= q025 & th <= q975, "Covered", "Missed")
    
    tibble(
      theta_true   = th,
      k_obs        = k_obs,
      theta_mean   = mean(draws),
      theta_median = median(draws),
      theta_q025   = q025,
      theta_q975   = q975,
      cover        = cover_lbl
    )
  })
}


# Make the recovery plot
plot_recovery <- function(results, n_trials) {
  ggplot(results, aes(x = theta_true, y = theta_median, color = cover)) +
    geom_errorbar(aes(ymin = theta_q025, ymax = theta_q975), width = 0) +
    geom_point(size = 1.5) +
    geom_abline(slope = 1, intercept = 0, linetype = 2) +
    scale_color_manual(
      values = c("Covered" = "skyblue", "Missed" = "red"),
      name = "CI covers true value"
    ) +
    labs(
      title = "Recovery of θ across true values",
      subtitle = sprintf("n = %d trials per data set", n_trials),
      x = expression(theta[true]),
      y = expression(hat(theta)~"(posterior median)")
    ) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    theme_classic(base_size = 12)
}

# Run
n_trials  <- 20
num_post  <- 8000
theta_grid <- seq(0.0, 1.0, by = 0.02)

results <- recover_theta_grid(theta_grid, n_trials, num_post, seed = 123)
print(results)

p_recovery <- plot_recovery(results, n_trials)
print(p_recovery)

