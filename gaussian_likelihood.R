rm(list = ls())

# =========================
# 0) Environment & Imports
# =========================
options(reticulate.use_uv = FALSE)
Sys.setenv(RETICULATE_PYTHON = "/Users/rjb779/Library/r-miniconda-arm64/envs/r-bf/bin/python")
Sys.setenv(KERAS_BACKEND = "jax")

suppressPackageStartupMessages({
  library(reticulate)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(tibble)
})

py_config()

.py <- function(fmt, ...) reticulate::py_run_string(sprintf(fmt, ...))

# Quiet Python/JAX logs and print versions (once)
py_run_string("
import os, numpy as np, bayesflow as bf
import keras, jax
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('KERAS_BACKEND', 'jax')
print('[ok] Backend   :', keras.config.backend())
print('[ok] Platform  :', jax.default_backend())
print('[ok] NumPy     :', np.__version__)
print('[ok] BayesFlow :', getattr(bf, '__version__', 'unknown'))
")


# ======================
# 1) Simulator (point x)
# ======================
# Likelihood-learning simulator: returns one 1-D x for each theta = (mu, sigma).
bf_build_simulator_gaussian_point <- function(
    mu_range    = c(-1.0, 1.0),
    sigma_range = c(0.3, 1.5)
) {
  stopifnot(length(mu_range) == 2L, length(sigma_range) == 2L)
  .py("
import numpy as np
mu_min, mu_max = %f, %f
s_min,  s_max  = %f, %f

def _as_int(n):
    try: return int(n)
    except (TypeError, ValueError):
        try: return int(n[0])
        except Exception as e:
            raise TypeError(f'Cannot coerce n={n!r} to int') from e

def _sample_theta(n):
    mu    = np.random.uniform(mu_min, mu_max, size=n).astype(np.float32)
    sigma = np.random.uniform(s_min,  s_max,  size=n).astype(np.float32)
    return np.stack([mu, sigma], axis=1).astype(np.float32)  # [n,2]

class GaussianPointSimulator:
    def sample(self, n):
        n = _as_int(n)
        theta = _sample_theta(n)                # [n,2]
        mu    = theta[:, 0:1]
        sigma = theta[:, 1:2]
        x = np.random.normal(loc=mu, scale=np.maximum(sigma, 1e-6)).astype(np.float32)  # [n,1]
        return {'theta': theta, 'x': x}

simulator = GaussianPointSimulator()
_test = simulator.sample((3,))
print('[sim ok] keys:', list(_test.keys()), '| theta', _test['theta'].shape, '| x', _test['x'].shape)
", mu_range[1], mu_range[2], sigma_range[1], sigma_range[2])
  invisible(TRUE)
}


# ==========================
# 2) Workflow (likelihood-NF)
# ==========================
# Builds a BayesFlow workflow that learns log p(x | theta).
bf_build_workflow_likelihood <- function(
    inference_net = c("coupling","diffusion"),
    lr = 1e-3
) {
  inference_net <- match.arg(inference_net)
  .py("
import bayesflow as bf
adapter = (bf.Adapter()
            .convert_dtype('float64','float32')
            .rename('theta', 'inference_conditions')
            .rename('x',     'inference_variables'))

if '%s' == 'coupling':
    inference_net = bf.networks.CouplingFlow(
        n_coupling_layers = 12,
        hidden_units      = [256,256,256]
    )
else:
    inference_net = bf.networks.DiffusionModel()

workflow = bf.BasicWorkflow(
    simulator              = simulator,
    adapter                = adapter,
    inference_network      = inference_net,
    inference_variables    = 'inference_variables',
    inference_conditions   = 'inference_conditions',
    initial_learning_rate  = %f,
    standardize            = None,      # keep x on original scale
    conditions_standardize = 'zscore'   # standardize theta only
)
_sim_obj = workflow.simulator
print('[lik-workflow] ok')
", inference_net, lr)
invisible(TRUE)
}


# ===========
# 3) Training
# ===========
bf_fit_online <- function(
    epochs = 40L,
    num_batches_per_epoch = 400L,
    batch_size = 400L,
    with_validation = TRUE,
    n_val = 800L
) {
  if (with_validation) {
    .py("val_raw = workflow.simulator.sample(%d)", as.integer(n_val))
    val_arg <- ", validation_data=val_raw"
  } else val_arg <- ""
  .py("history = workflow.fit_online(epochs=%d, batch_size=%d, num_batches_per_epoch=%d%s)",
      as.integer(epochs), as.integer(batch_size), as.integer(num_batches_per_epoch), val_arg)
  invisible(TRUE)
}


# ======================================
# 4) Analytic & Learned Likelihood Tools
# ======================================
# Analytic log p(X | mu, sigma) for a set X = {x_1..x_n}
analytic_loglik_gaussian_set <- function(x, mu, sigma, eps = 1e-8) {
  stopifnot(is.numeric(x), length(x) >= 1L)
  s  <- pmax(as.numeric(sigma), eps)
  nT <- length(x)
  -0.5 * nT * (log(2*pi) + 2*log(s)) - sum((x - mu)^2) / (2*s^2)
}

# Vectorized learned set log-likelihood on a theta grid (robust & chunked)
bf_eval_likelihood_on_grid <- function(theta_grid, x_set, chunk = 20000L) {
  stopifnot(is.matrix(theta_grid) && ncol(theta_grid) == 2)
  stopifnot(is.numeric(x_set) && length(x_set) >= 1L)
  stopifnot(chunk >= 1L)
  py_run_string("
import numpy as _np
def _lik_grid_chunked(theta_grid_np, x_vec_np, chunk=20000):
    theta_grid_np = _np.asarray(theta_grid_np, dtype=_np.float32)   # [N,2]
    x_vec_np      = _np.asarray(x_vec_np,      dtype=_np.float32)   # [T]
    N = int(theta_grid_np.shape[0]); T = int(x_vec_np.shape[0])
    ll = _np.empty((N,), dtype=_np.float64)
    start = 0
    while start < N:
        end = min(start + int(chunk), N)
        n_chunk = end - start
        TH = _np.repeat(theta_grid_np[start:end], T, axis=0)               # [n_chunk*T,2]
        X  = _np.repeat(x_vec_np.reshape(-1,1), n_chunk, axis=1).T
        X  = X.reshape(-1,1).astype(_np.float32)
        lp = workflow.log_prob(data={'x': X, 'theta': TH})                 # [n_chunk*T]
        ll[start:end] = lp.reshape(n_chunk, T).sum(axis=1)
        start = end
    return ll
")
  as.numeric(py$`_lik_grid_chunked`(theta_grid, as.numeric(x_set), as.integer(chunk)))
}

# Training support box (as used by the simulator)
bf_get_theta_support <- function() {
  py <- reticulate::py
  get_or <- function(name, default) if (!is.null(py[[name]])) as.numeric(py[[name]]) else default
  c(mu_min = get_or("mu_min", -1.0), mu_max = get_or("mu_max", 1.0),
    s_min  = get_or("s_min",   0.3), s_max  = get_or("s_max",   1.5))
}


# ======================
# 5) Validation Routines
# ======================
# Pointwise (per-observation) checks
bulk_check_points <- function(n = 5000L, seed = 123) {
  set.seed(seed)
  support <- bf_get_theta_support()
  mu    <- runif(n, support["mu_min"], support["mu_max"])
  sigma <- runif(n, support["s_min"],  support["s_max"])
  x     <- rnorm(n, mu, sigma)
  ll_an <- -0.5*(log(2*pi) + 2*log(sigma)) - ((x - mu)^2)/(2*sigma^2)
  py_run_string("
import numpy as _np
def _logprob_many(mu, sigma, x):
    mu    = _np.asarray(mu,    dtype=_np.float32).reshape(-1,1)
    sigma = _np.asarray(sigma, dtype=_np.float32).reshape(-1,1)
    x     = _np.asarray(x,     dtype=_np.float32).reshape(-1,1)
    TH = _np.concatenate([mu, sigma], axis=1)
    return workflow.log_prob(data={'x': x, 'theta': TH})
")
  ll_le <- as.numeric(py$`_logprob_many`(mu, sigma, x))
  df <- tibble(mu, sigma, x, ll_analytic = ll_an, ll_learned = ll_le, diff = ll_le - ll_an,
               abs_z = abs((x - mu)/sigma))
  fit <- lm(ll_learned ~ ll_analytic, data = df)
  list(
    metrics = list(
      intercept = unname(coef(fit)[1]),
      slope     = unname(coef(fit)[2]),
      r2        = summary(fit)$r.squared,
      rmse      = sqrt(mean(df$diff^2)),
      bias      = mean(df$diff)
    ),
    plots = list(
      scatter = ggplot(df, aes(ll_analytic, ll_learned)) +
        geom_point(alpha=.4, size=2, color = "skyblue") +
        geom_abline(slope=1, intercept=0, linetype=2, color = "black") +
        labs(title="Per-trial log p(x|θ)", x="Analytic", y="Learned") +
        theme_classic(),
      resid_vs_sigma = ggplot(df, aes(sigma, diff)) +
        geom_hline(yintercept=0, linetype=2) +
        geom_point(alpha=.15, size=.9) +
        labs(title="Per-trial residual vs σ", x="σ", y="Residual (learned−analytic)") +
        theme_classic(),
      resid_vs_absz = ggplot(df, aes(abs_z, diff)) +
        geom_hline(yintercept=0, linetype=2) +
        geom_point(alpha=.15, size=.9) +
        labs(title="Per-trial residual vs |z|", x="|x−μ|/σ", y="Residual") +
        theme_classic()
    )
  )
}

# Per-set (summed over observations) checks + per-trial equivalents
bulk_check_sets <- function(x_obs, n_sets = 2000L, seed = 123) {
  set.seed(seed)
  support <- bf_get_theta_support()
  mu    <- runif(n_sets, support["mu_min"], support["mu_max"])
  sigma <- runif(n_sets, support["s_min"],  support["s_max"])
  theta_grid <- cbind(mu, sigma)
  ll_le <- bf_eval_likelihood_on_grid(theta_grid, x_obs)
  ll_an <- mapply(function(m, s) analytic_loglik_gaussian_set(x_obs, m, s), mu, sigma)
  df <- tibble(mu, sigma, ll_analytic = ll_an, ll_learned = ll_le, diff = ll_le - ll_an)
  fit <- lm(ll_learned ~ ll_analytic, data = df)
  nT  <- length(x_obs)
  list(
    metrics = list(
      intercept     = unname(coef(fit)[1]),
      slope         = unname(coef(fit)[2]),
      r2            = summary(fit)$r.squared,
      rmse_set      = sqrt(mean(df$diff^2)),
      bias_set      = mean(df$diff),
      rmse_per_trial= sqrt(mean(df$diff^2)) / nT,
      bias_per_trial= mean(df$diff) / nT,
      n_trials      = nT
    ),
    plots = list(
      scatter = ggplot(df, aes(ll_analytic, ll_learned)) +
        geom_point(alpha=.4, size=2, color = "skyblue") +
        geom_abline(slope=1, intercept=0, linetype=2, color = "black") +
        labs(title=sprintf("Set-summed log p(X|θ) (n=%d)", nT),
             x="Analytic", y="Learned") +
        theme_classic(),
      resid = ggplot(df, aes(ll_analytic, diff)) +
        geom_hline(yintercept=0, linetype=2) +
        geom_point(alpha=.2, size=.9) +
        labs(title="Set-summed log p(X|θ)", x="Analytic", y="Residual") +
        theme_classic()
    )
  )
}


# =====================
# 6) Diagnostic Slices
# =====================
bf_plot_slice_over_mu <- function(x_obs,
                                  mu_seq = seq(-1, 1, length.out = 200),
                                  sigmas = c(0.4, 0.7, 1.0),
                                  normalise_each_slice = TRUE) {
  grid <- expand.grid(mu = mu_seq, sigma = sigmas)
  theta_grid <- as.matrix(grid[, c("mu","sigma")])
  ll_le <- bf_eval_likelihood_on_grid(theta_grid, x_obs)
  ll_an <- mapply(function(m, s) analytic_loglik_gaussian_set(x_obs, m, s), grid$mu, grid$sigma)
  df <- tibble(mu = grid$mu, sigma = factor(grid$sigma), learned = ll_le, analytic = ll_an)
  if (normalise_each_slice) {
    df <- df |>
      group_by(sigma) |>
      mutate(learned = learned - max(learned), analytic = analytic - max(analytic)) |>
      ungroup()
  }
  df |>
    pivot_longer(c("learned","analytic"), names_to = "source", values_to = "loglik") |>
    ggplot(aes(mu, loglik, colour = sigma, linetype = source)) +
    geom_line(linewidth = 0.9) +
    labs(title = "Log-likelihood slices over μ (fixed σ)", x = expression(mu), y = "log p(X | μ, σ)") +
    theme_classic()
}

bf_plot_slice_over_sigma <- function(x_obs,
                                     sig_seq = seq(0.3, 1.5, length.out = 200),
                                     mus = c(-0.4, 0.0, 0.4),
                                     normalise_each_slice = TRUE) {
  grid <- expand.grid(mu = mus, sigma = sig_seq)
  theta_grid <- as.matrix(grid[, c("mu","sigma")])
  ll_le <- bf_eval_likelihood_on_grid(theta_grid, x_obs)
  ll_an <- mapply(function(m, s) analytic_loglik_gaussian_set(x_obs, m, s), grid$mu, grid$sigma)
  df <- tibble(sigma = grid$sigma, mu = factor(grid$mu), learned = ll_le, analytic = ll_an)
  if (normalise_each_slice) {
    df <- df |>
      group_by(mu) |>
      mutate(learned = learned - max(learned), analytic = analytic - max(analytic)) |>
      ungroup()
  }
  df |>
    pivot_longer(c("learned","analytic"), names_to = "source", values_to = "loglik") |>
    ggplot(aes(sigma, loglik, colour = mu, linetype = source)) +
    geom_line(linewidth = 0.9) +
    labs(title = "Log-likelihood slices over σ (fixed μ)", x = expression(sigma), y = "log p(X | μ, σ)") +
    theme_classic()
}


# =================
# 7) End-to-end Run
# =================
# Build simulator + workflow
bf_build_simulator_gaussian_point(mu_range = c(-1.0, 1.0), sigma_range = c(0.3, 1.5))
bf_build_workflow_likelihood(inference_net = "coupling", lr = 1e-3)

# Train
bf_fit_online(epochs = 30, num_batches_per_epoch = 400, batch_size = 400,
              with_validation = TRUE, n_val = 800)

# Quick pointwise check
pt <- bulk_check_points(n = 10000)
pt$metrics
print(pt$plots$scatter)
# print(pt$plots$resid_vs_sigma)
# print(pt$plots$resid_vs_absz)

# Make an observed set and do set-wise checks
set.seed(42)
n_trials <- 500
mu0 <- 0.0; sigma0 <- 0.5
x_obs <- rnorm(n_trials, mu0, sigma0)

set_chk <- bulk_check_sets(x_obs, n_sets = 10000)
set_chk$metrics
print(set_chk$plots$scatter)
# print(set_chk$plots$resid)

# ---- Combine per-trial and set-summed scatter plots with patchwork ----
suppressPackageStartupMessages(library(patchwork))

# 1) Pull the two ggplots
p_trial <- pt$plots$scatter +
  labs(title = "Per-trial log p(x|θ)",
       x = "Analytic", y = "Learned") +
  theme(legend.position = "none")

p_set <- set_chk$plots$scatter +
  labs(title = sprintf("Set-summed log p(X|θ) (n=%d)",
                       set_chk$metrics$n_trials),
       x = "Analytic", y = "Learned")

# 3) R-squared annotations (upper-left of each panel)
r2_trial <- pt$metrics$r2
r2_set   <- set_chk$metrics$r2

add_r2 <- function(p, r2, size = 3.6) {
  p + annotate(
    "text",
    x = -Inf, y = Inf,
    label = sprintf("R\u00B2 = %.3f", r2),
    hjust = -0.1, vjust = 1.1, size = size
  )
}

p_trial <- add_r2(p_trial, r2_trial)
p_set   <- add_r2(p_set,   r2_set)

# 4) Combine with patchwork
p_both <- (p_trial | p_set) +
  plot_annotation(
    title = "Learned vs Analytic Log-Likelihood",
    theme = theme(plot.title = element_text(hjust = 0.5)),
    tag_levels = "A"
  ) &
  theme_classic(base_size = 11)

print(p_both)

# Save
dir.create("plots", showWarnings = FALSE)
# Vector PDF with Cairo (handles Unicode)
ggplot2::ggsave("plots/lik_scatter_combo.pdf", p_both,
                width = 9, height = 4.5, device = cairo_pdf)

# Slices for diagnostics
p_mu <- bf_plot_slice_over_mu(x_obs, sigmas = c(0.4, 0.7, 1.0))
p_sg <- bf_plot_slice_over_sigma(x_obs, mus = c(-0.3, 0.0, 0.3))
print(p_mu)
print(p_sg)
