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
# 1) Simulator (DDM x=2)
# ======================
# Returns ONE trial per theta: x = [rt, resp], with resp ∈ {+1, -1}.
bf_build_simulator_ddm_point <- function(
    a_range = c(0.7, 2.5),
    v_range = c(-3.0, 3.0),
    t0_range = c(0.20, 0.60),
    dt = 0.002,        # Euler Δt (s)
    tmax = 5.0         # hard cap on decision time (s)
) {
  stopifnot(length(a_range)==2L, length(v_range)==2L, length(t0_range)==2L)
  .py("
import numpy as np, math
a_lo, a_hi   = %f, %f
v_lo, v_hi   = %f, %f
t0_lo, t0_hi = %f, %f
_dt   = %f
_tmax = %f
rng = np.random.default_rng()

def _as_int(n):
    try: return int(n)
    except (TypeError, ValueError):
        try: return int(n[0])
        except Exception as e:
            raise TypeError(f'Cannot coerce n={n!r} to int') from e

def _sample_theta(n):
    a  = rng.uniform(a_lo,  a_hi,  size=n).astype(np.float32)
    v  = rng.uniform(v_lo,  v_hi,  size=n).astype(np.float32)
    t0 = rng.uniform(t0_lo, t0_hi, size=n).astype(np.float32)
    return np.stack([a, v, t0], axis=1).astype(np.float32)  # [n,3]

def _simulate_one(a, v, t0):
    half_a = a/2.0
    max_steps = int(_tmax/_dt)
    x = 0.0
    for step in range(max_steps):
        x += v*_dt + math.sqrt(_dt)*rng.standard_normal()
        if x >=  half_a: return (step+1)*_dt + t0, +1.0
        if x <= -half_a: return (step+1)*_dt + t0, -1.0
    # timeout fallback
    return _tmax + t0, (+1.0 if x>=0.0 else -1.0)

class DDMPointSimulator:
    def sample(self, n):
        n = _as_int(n)
        theta = _sample_theta(n)              # [n,3]
        X = np.empty((n,2), dtype=np.float32)
        for i in range(n):
            a, v, t0 = map(float, theta[i])
            rt, r = _simulate_one(a, v, t0)
            X[i,0] = rt
            X[i,1] = r       # +1 or -1
        return {'theta': theta, 'x': X}

simulator = DDMPointSimulator()
_test = simulator.sample((3,))
print('[sim ok DDM] keys:', list(_test.keys()), '| theta', _test['theta'].shape, '| x', _test['x'].shape)
", a_range[1], a_range[2], v_range[1], v_range[2], t0_range[1], t0_range[2], dt, tmax)
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

bf_get_theta_support_ddm <- function() {
  py <- reticulate::py
  get_or <- function(name, default) if (!is.null(py[[name]])) as.numeric(py[[name]]) else default
  c(a_lo = get_or("a_lo", 0.5), a_hi = get_or("a_hi", 2.5),
    v_lo = get_or("v_lo",-2.0), v_hi = get_or("v_hi", 2.0),
    t0_lo= get_or("t0_lo",0.20),t0_hi= get_or("t0_hi",0.40))
}

suppressPackageStartupMessages(require(rtdists))

# Vectorized analytic per-trial log p(x | a,v,t0) using rtdists::ddiffusion
# x is a matrix/data.frame with columns: rt (seconds), resp (+1 / -1)
analytic_loglik_ddm_points <- function(x, a, v, t0, z = 0.5, s = 1) {
  stopifnot(is.matrix(x) || is.data.frame(x))
  rt   <- as.numeric(x[,1])
  resp <- as.numeric(x[,2])
  resp_chr <- ifelse(resp > 0, "upper", "lower")
  dens <- rtdists::ddiffusion(rt,
                              response = resp_chr,
                              a = a, v = v, t0 = t0,
                              z = z * a,  # z is absolute; rtdists expects absolute (not proportion)
                              s = s,
                              # you can tweak the accuracy/speed controls if needed:
                              # precision = 3e-6
  )
  log(pmax(dens, .Machine$double.xmin))
}

# Analytic set log-likelihood: sum over rows in x_set
analytic_loglik_ddm_set <- function(x_set, a, v, t0, z = 0.5, s = 1) {
  sum(analytic_loglik_ddm_points(x_set, a, v, t0, z = z, s = s))
}

# x_set: matrix/data.frame with columns (rt, resp ∈ {+1,-1})
# theta_grid: [N,3] with columns (a, v, t0)
bf_eval_likelihood_on_grid_ddm <- function(theta_grid, x_set, chunk = 20000L) {
  stopifnot(is.matrix(theta_grid) && ncol(theta_grid) == 3)
  stopifnot(nrow(x_set) >= 1L)
  x_mat <- as.matrix(x_set)
  py_run_string("
import numpy as _np
def _lik_grid_chunked_ddm(theta_grid_np, x_mat_np, chunk=20000):
    TH = _np.asarray(theta_grid_np, dtype=_np.float32)   # [N,3]
    X  = _np.asarray(x_mat_np,    dtype=_np.float32)     # [T,2]
    N, T = TH.shape[0], X.shape[0]
    ll = _np.empty((N,), dtype=_np.float64)
    start = 0
    while start < N:
        end = min(start + int(chunk), N)
        n_chunk = end - start
        THrep = _np.repeat(TH[start:end], T, axis=0)           # [n_chunk*T,3]
        Xrep  = _np.tile(X, (n_chunk, 1))                      # [n_chunk*T,2]
        lp = workflow.log_prob(data={'x': Xrep, 'theta': THrep})  # [n_chunk*T]
        ll[start:end] = lp.reshape(n_chunk, T).sum(axis=1)
        start = end
    return ll
")
  as.numeric(py$`_lik_grid_chunked_ddm`(theta_grid, x_mat, as.integer(chunk)))
}

# Per-trial check: draw many (theta, x) from the simulator and compare log-liks
bulk_check_points_ddm <- function(n = 8000L, seed = 123) {
  set.seed(seed)
  .py("batch = simulator.sample(%d)", as.integer(n))
  th <- py$batch$theta  # [n,3]
  x  <- py$batch$x      # [n,2]
  
  a <- as.numeric(th[,1]); v <- as.numeric(th[,2]); t0 <- as.numeric(th[,3])
  rt <- as.numeric(x[,1]); r  <- as.numeric(x[,2])
  x_df <- cbind(rt, r)
  
  # analytic
  ll_an <- analytic_loglik_ddm_points(x_df, a, v, t0)
  
  # learned
  py_run_string("
import numpy as _np
def _logprob_many_ddm(th_np, x_np):
    th_np = _np.asarray(th_np, dtype=_np.float32)
    x_np  = _np.asarray(x_np,  dtype=_np.float32)
    return workflow.log_prob(data={'x': x_np, 'theta': th_np})
")
  ll_le <- as.numeric(py$`_logprob_many_ddm`(th, x))
  
  df <- tibble(a, v, t0, rt, resp = r,
               ll_analytic = ll_an, ll_learned = ll_le,
               diff = ll_le - ll_an)
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
        geom_point(alpha=.35, size=1.6, color="skyblue") +
        geom_abline(slope=1, intercept=0, linetype=2) +
        labs(title = "Per-trial log p(x|θ): DDM", x="Analytic", y="Learned") +
        theme_classic(),
      resid_vs_rt = ggplot(df, aes(rt, diff)) +
        geom_hline(yintercept=0, linetype=2) +
        geom_point(alpha=.15, size=.8) +
        labs(title="Per-trial residual vs RT", x="RT (s)", y="Residual") +
        theme_classic()
    )
  )
}

# Per-set check: fix one observed set X and compare learned vs analytic across θ-grid
bulk_check_sets_ddm <- function(x_obs, n_sets = 4000L, seed = 123) {
  set.seed(seed)
  sup <- bf_get_theta_support_ddm()
  a  <- runif(n_sets, sup['a_lo'],  sup['a_hi'])
  v  <- runif(n_sets, sup['v_lo'],  sup['v_hi'])
  t0 <- runif(n_sets, sup['t0_lo'], sup['t0_hi'])
  theta_grid <- cbind(a, v, t0)
  
  ll_le <- bf_eval_likelihood_on_grid_ddm(theta_grid, x_obs)
  ll_an <- mapply(function(aa, vv, tt) analytic_loglik_ddm_set(x_obs, aa, vv, tt), a, v, t0)
  
  df <- tibble(a, v, t0, ll_analytic = ll_an, ll_learned = ll_le, diff = ll_le - ll_an)
  fit <- lm(ll_learned ~ ll_analytic, data = df)
  nT  <- nrow(x_obs)
  
  list(
    metrics = list(
      intercept      = unname(coef(fit)[1]),
      slope          = unname(coef(fit)[2]),
      r2             = summary(fit)$r.squared,
      rmse_set       = sqrt(mean(df$diff^2)),
      bias_set       = mean(df$diff),
      rmse_per_trial = sqrt(mean(df$diff^2)) / nT,
      bias_per_trial = mean(df$diff) / nT,
      n_trials       = nT
    ),
    plots = list(
      scatter = ggplot(df, aes(ll_analytic, ll_learned)) +
        geom_point(alpha=.4, size=1.6, color="skyblue") +
        geom_abline(slope=1, intercept=0, linetype=2) +
        labs(title=sprintf("Set-summed log p(X|θ): DDM (n=%d)", nT),
             x="Analytic", y="Learned") +
        theme_classic(),
      resid = ggplot(df, aes(ll_analytic, diff)) +
        geom_hline(yintercept=0, linetype=2) +
        geom_point(alpha=.2, size=.8) +
        labs(title="Set-summed residuals", x="Analytic", y="Learned − Analytic") +
        theme_classic()
    )
  )
}


# =================
# 7) End-to-end Run
# =================
# Build simulator + workflow
bf_build_simulator_ddm_point(
  a_range = c(0.5, 1.5),
  v_range = c(-1.5, 1.5),
  t0_range = c(0.20, 0.40)
)

bf_build_workflow_likelihood(inference_net = "coupling", lr = 1e-3)

# Train the learned likelihood
bf_fit_online(
  epochs = 30,
  num_batches_per_epoch = 300,
  batch_size = 384,
  with_validation = TRUE,
  n_val = 800
)

# ---- Per-trial diagnostic ----
pt <- bulk_check_points_ddm(n = 10000)
pt$metrics
print(pt$plots$scatter)
# print(pt$plots$resid_vs_rt)

# ---- Build one observed set X and compare over a θ-grid ----
set.seed(42)
n_trials <- 1200
# simulate one X via the simulator so train/test are consistent:
.py(sprintf("
import numpy as np
def _simulate_set_for(a, v, t0, nT=%d):
    X = np.empty((nT,2), dtype=np.float32)
    for i in range(nT):
        rt, r = _simulate_one(float(a), float(v), float(t0))
        X[i,0] = rt; X[i,1] = r
    return X
", n_trials))
a0 <- 1.4; v0 <- 1.0; t00 <- 0.35
x_obs <- py$`_simulate_set_for`(a0, v0, t00)

set_chk <- bulk_check_sets_ddm(x_obs, n_sets = 8000)
set_chk$metrics
print(set_chk$plots$scatter)
# print(set_chk$plots$resid)

# ---- Combined learned-vs-analytic panels ----
suppressPackageStartupMessages(library(patchwork))
r2_trial <- pt$metrics$r2; r2_set <- set_chk$metrics$r2
add_r2 <- function(p, r2, size=3.6) p + annotate("text", x=-Inf, y=Inf,
                                                 label=sprintf("R\u00B2 = %.3f", r2),
                                                 hjust=-0.1, vjust=1.1, size=size)
p_both <- (add_r2(pt$plots$scatter, r2_trial) | add_r2(set_chk$plots$scatter, r2_set)) +
  plot_annotation(title="DDM Learned vs Analytic Log-Likelihood",
                  theme = theme(plot.title = element_text(hjust = 0.5)))
print(p_both)

dir.create("plots", showWarnings = FALSE)
ggplot2::ggsave("plots/ddm_lik_scatter_combo.pdf", p_both,
                width = 9, height = 4.5, device = cairo_pdf)
