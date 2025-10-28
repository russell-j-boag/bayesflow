rm(list = ls())

# =======================
# 0) Environment / setup
# =======================
options(reticulate.use_uv = FALSE)
Sys.setenv(RETICULATE_PYTHON = "/Users/rjb779/Library/r-miniconda-arm64/envs/r-bf/bin/python")
Sys.setenv(KERAS_BACKEND = "jax")

suppressPackageStartupMessages({
  library(reticulate)
})

py_config()

# Small helper: sprintf + py_run_string
.bf_py <- function(fmt, ...) reticulate::py_run_string(sprintf(fmt, ...))

# One-time Python warmup / versions
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


# ========================
# 1) Simulator (Gaussian)
# ========================
# Simulates SETS (nT obs per case): ideal for DeepSet summary encoders.
bf_build_simulator_gaussian_set <- function(
    n_trials          = 400L,
    mu_range          = c(-1.0, 1.0),
    sigma_range       = c(0.3, 1.5),
    standardize_x     = FALSE,
    expose_split_keys = FALSE
) {
  stopifnot(length(mu_range) == 2L, length(sigma_range) == 2L)
  nT <- as.integer(n_trials)  # <— was T
  ex <- if (expose_split_keys) "True" else "False"
  
  reticulate::py_run_string(sprintf("
import numpy as np

_T             = %d
mu_min, mu_max = %f, %f
s_min,  s_max  = %f, %f
standardize_x  = %s
expose_split   = %s

def _as_int(n):
    try: return int(n)
    except (TypeError, ValueError):
        try: return int(n[0])
        except Exception as e:
            raise TypeError(f'Cannot coerce n={n} to int') from e

def _sample_params(n):
    mu    = np.random.uniform(mu_min, mu_max, size=n).astype(np.float32)
    sigma = np.random.uniform(s_min,  s_max,  size=n).astype(np.float32)
    return np.stack([mu, sigma], axis=1)  # [n, 2]

def _simulate_case(theta_row):
    mu, sigma = float(theta_row[0]), float(theta_row[1])
    sigma = max(sigma, 1e-6)
    x = np.random.normal(loc=mu, scale=sigma, size=_T).astype(np.float32)   # [T]
    if standardize_x:
        m, s = x.mean(), x.std()
        if s > 1e-9: x = (x - m) / s
    return x[:, None]  # [T, 1]

class GaussianSetSimulator:
    def __init__(self, T): self.T = T
    def sample(self, n):
        n = _as_int(n)
        theta = _sample_params(n)                 # [n, 2]
        X = np.empty((n, self.T, 1), dtype=np.float32)
        for i in range(n): X[i] = _simulate_case(theta[i])
        return {'theta': theta.astype(np.float32), 'trials': X}

class KeySplitSimulator:
    def __init__(self, base): self.base = base; self.T = getattr(base, 'T', None)
    def sample(self, n):
        d  = self.base.sample(n)
        th = d['theta']
        d['mu']    = th[:, 0].astype('float32')
        d['sigma'] = th[:, 1].astype('float32')
        return d

simulator = GaussianSetSimulator(_T)
if expose_split:
    simulator = KeySplitSimulator(simulator)

_test = simulator.sample(2)
print('[sim ok] keys:', list(_test.keys()), '| theta', _test['theta'].shape, '| trials', _test['trials'].shape)
", nT, mu_range[1], mu_range[2], sigma_range[1], sigma_range[2],
                                    if (standardize_x) "True" else "False", ex))
  invisible(TRUE)
}


# ==========================
# 2) Workflow (DeepSet + NF)
# ==========================
bf_build_workflow_deepset <- function(
    phi_units     = c(128, 128),
    rho_units     = c(128, 128),
    aggregation   = c("mean","sum","max"),
    flow_type     = c("coupling","autoregressive"),
    n_flow_layers = 6L,
    flow_hidden   = c(128, 128),
    param_keys    = "theta",
    data_key      = "trials"
) {
  aggregation <- match.arg(aggregation)
  flow_type   <- match.arg(flow_type)
  
  phi_py  <- paste0("[", paste(as.integer(phi_units), collapse=","), "]")
  rho_py  <- paste0("[", paste(as.integer(rho_units), collapse=","), "]")
  flow_py <- paste0("[", paste(as.integer(flow_hidden), collapse=","), "]")
  
  param_keys_vec <- as.character(param_keys)
  py_param_list  <- paste0("[", paste(sprintf("'%s'", param_keys_vec), collapse=", "), "]")
  use_concat     <- length(param_keys_vec) > 1L
  data_key_py    <- sprintf("'%s'", as.character(data_key))
  
  # Ensure Python-side simulator exists; wrap if needed
  py_run_string("
import bayesflow as bf, numpy as np, builtins
if 'simulator' not in globals() and not hasattr(builtins, 'simulator'):
    raise RuntimeError('Python `simulator` is not defined.')

class CallableSimulator:
    def __init__(self, fn): self.fn = fn
    def sample(self, n):    return self.fn(n)

if hasattr(simulator, 'sample') and callable(getattr(simulator, 'sample')):
    _sim_obj = simulator
elif callable(simulator):
    _sim_obj = CallableSimulator(simulator)
else:
    raise TypeError('`simulator` must be callable or have a `.sample(n)` method.')
")
  
  # Adapter
  adapter_block <- if (use_concat) {
    paste(
      "(",
      "bf.Adapter()",
      "  .convert_dtype('float64','float32')",
      sprintf("  .concatenate(%s, into='inference_variables')", py_param_list),
      sprintf("  .rename(%s, 'summary_variables')", data_key_py),
      ")",
      sep = "\n"
    )
  } else {
    paste(
      "(",
      "bf.Adapter()",
      "  .convert_dtype('float64','float32')",
      sprintf("  .rename('%s', 'inference_variables')", param_keys_vec[[1]]),
      sprintf("  .rename(%s,   'summary_variables')", data_key_py),
      ")",
      sep = "\n"
    )
  }
  
  # Build networks + workflow
  py_code <- paste(
    "import numpy as np, bayesflow as bf",
    sprintf("adapter = %s", adapter_block),
    sprintf("summary_net = bf.networks.DeepSet(phi_units=%s, rho_units=%s, aggregation='%s')",
            phi_py, rho_py, aggregation),
    "try:",
    if (flow_type == "coupling") {
      sprintf("    inference_net = bf.networks.CouplingFlow(n_coupling_layers=%d, hidden_units=%s)",
              as.integer(n_flow_layers), flow_py)
    } else {
      sprintf("    inference_net = bf.networks.AutoregressiveFlow(n_layers=%d, hidden_units=%s)",
              as.integer(n_flow_layers), flow_py)
    },
    "except TypeError as e:",
    "    print('[warn] Flow kwargs not supported; using defaults:', e)",
    if (flow_type == "coupling") "    inference_net = bf.networks.CouplingFlow()" else
      "    inference_net = bf.networks.AutoregressiveFlow()",
    "workflow = bf.BasicWorkflow(",
    "    simulator = _sim_obj,",
    "    adapter   = adapter,",
    "    inference_network  = inference_net,",
    "    summary_network    = summary_net,",
    "    inference_variables = 'inference_variables',",
    "    summary_variables   = 'summary_variables'",
    ")",
    "_raw = _sim_obj.sample(4)",
    "_bat = adapter(_raw)",
    "shapes = {k: (np.array(v).shape if hasattr(v, '__array__') else getattr(v, 'shape', None))",
    "          for k, v in _bat.items()}",
    "print('[ok] workflow ready; adapted shapes:', shapes)",
    sep = "\n"
  )
  
  py_run_string(py_code)
  invisible(TRUE)
}


# ===========
# 3) Training
# ===========
bf_fit_online <- function(
    epochs = 3L,
    num_batches_per_epoch = 80L,
    batch_size = 200L,
    with_validation = TRUE,
    n_val = 400L
) {
  if (with_validation) {
    .bf_py("val_raw = _sim_obj.sample(%d)", as.integer(n_val))
    val_arg <- ", validation_data=val_raw"
  } else {
    val_arg <- ""
  }
  .bf_py("history = workflow.fit_online(epochs=%d, batch_size=%d, num_batches_per_epoch=%d%s)",
         as.integer(epochs), as.integer(batch_size), as.integer(num_batches_per_epoch), val_arg)
  invisible(TRUE)
}


# ==================
# 4) Posterior draws
# ==================
bf_sample_posteriors <- function(n_test = 200L, n_draws = 2000L) {
  .bf_py("
import numpy as np, pandas as pd

_raw = _sim_obj.sample(%d)

if   'trials' in _raw: data_key = 'trials'
elif 'x'      in _raw: data_key = 'x'
else: raise KeyError('Expected a data key `trials` or `x` in simulator output.')

X = _raw[data_key]                    # [N, T, D]
N = X.shape[0]
_samples = workflow.sample(num_samples=%d, conditions={data_key: X})

# Extract posterior array as (draw, case, param)
S = None
if isinstance(_samples, dict):
    for k in ['inference_variables', 'posterior', 'theta', 'z', 'y']:
        if k in _samples and isinstance(_samples[k], np.ndarray):
            S = _samples[k]; break
    if S is None:
        parts = [v for v in _samples.values() if isinstance(v, np.ndarray)]
        if parts: S = np.concatenate(parts, axis=-1)
elif isinstance(_samples, (list, tuple)):
    for v in _samples:
        if isinstance(v, np.ndarray): S = v; break
elif isinstance(_samples, np.ndarray):
    S = _samples

if S is None: raise ValueError('Could not locate posterior samples array.')

if S.ndim != 3: raise ValueError(f'Expect (draw, case, param)-like array, got {S.shape}')

# Axis normalization
shape = S.shape
draw_axis = shape.index(%d) if %d in shape else 0
case_axis = shape.index(N)  if N  in shape else (1 if draw_axis==0 else 0)
param_axis = [ax for ax in (0,1,2) if ax not in (draw_axis, case_axis)][0]
if (draw_axis, case_axis, param_axis) != (0,1,2):
    S = np.transpose(S, (draw_axis, case_axis, param_axis))

DRAWS, CASES, DPAR = S.shape
flat = S.reshape(DRAWS*CASES, DPAR)
df = pd.DataFrame(flat, columns=[f'p{j}' for j in range(DPAR)])
df['draw'] = np.repeat(np.arange(DRAWS), CASES)
df['case'] = np.tile(np.arange(CASES), DRAWS)

theta_true = None
for k in ['theta', 'inference_variables', 'params', 'prior']:
    if k in _raw: theta_true = _raw[k]; break

if isinstance(theta_true, np.ndarray):
    if theta_true.ndim == 1: theta_true = theta_true.reshape(N, 1)
    th_flat = np.vstack([theta_true for _ in range(DRAWS)])  # stack by draws
    for j in range(theta_true.shape[1]): df[f'true_p{j}'] = th_flat[:, j]

bf_last_samples_df = df
print('[ok] posterior table:', bf_last_samples_df.shape)
", as.integer(n_test), as.integer(n_draws), as.integer(n_draws), as.integer(n_draws))
  
  out <- py$bf_last_samples_df
  if (requireNamespace("tibble", quietly = TRUE)) tibble::as_tibble(out) else out
}


# ==========================
# 5) Summaries & conversion
# ==========================
bf_as_draws_array <- function(df) {
  stopifnot(is.data.frame(df), all(c("draw","case") %in% names(df)))
  pcols <- grep("^p\\d+$", names(df), value = TRUE)
  stopifnot(length(pcols) > 0)
  
  df <- df[order(df$draw, df$case), c("draw","case", pcols), drop = FALSE]
  M  <- as.matrix(df[, pcols, drop = FALSE])         # (D*C) x K
  D  <- length(unique(df$draw))
  C  <- length(unique(df$case))
  K  <- length(pcols)
  
  arr <- array(NA_real_, dim = c(D, C, K))
  for (j in seq_len(K)) arr[,,j] <- matrix(M[, j], nrow = D, ncol = C, byrow = TRUE)
  dimnames(arr) <- list(NULL, case = seq_len(C) - 1L, param = pcols)
  arr
}

bf_extract_truth <- function(df) {
  tcols <- grep("^true_p\\d+$", names(df), value = TRUE)
  if (!length(tcols)) return(NULL)
  one <- df[df$draw == min(df$draw), c("case", tcols), drop = TRUE]
  one <- one[order(one$case), , drop = FALSE]
  as.matrix(one[, tcols, drop = FALSE])   # [C, K_true]
}

bf_summarise_draws <- function(x, y_true = NULL, param_names = NULL) {
  if (is.data.frame(x)) {
    df <- x
    A  <- bf_as_draws_array(df)
    if (is.null(y_true)) y_true <- bf_extract_truth(df)
  } else {
    A <- x
  }
  stopifnot(length(dim(A)) == 3L)
  D <- dim(A)[1]; C <- dim(A)[2]; K <- dim(A)[3]
  
  pnames <- dimnames(A)[[3]]
  if (!is.null(param_names)) { stopifnot(length(param_names) == K); pnames <- param_names }
  
  # summaries over draws
  mean_mat   <- apply(A, c(2,3), mean)
  median_mat <- apply(A, c(2,3), median)
  q025_mat   <- apply(A, c(2,3), quantile, probs = 0.025)
  q975_mat   <- apply(A, c(2,3), quantile, probs = 0.975)
  sd_mat     <- apply(A, c(2,3), sd)
  
  out <- do.call(rbind, lapply(seq_len(C), function(ci) {
    data.frame(
      case   = ci - 1L,
      param  = pnames,
      mean   = mean_mat[ci, ],
      median = median_mat[ci, ],
      q025   = q025_mat[ci, ],
      q975   = q975_mat[ci, ],
      sd     = sd_mat[ci, ],
      row.names = NULL
    )
  }))
  
  if (!is.null(y_true)) {
    if (is.vector(y_true)) y_true <- matrix(rep(y_true, each = C), nrow = C, byrow = TRUE)
    stopifnot(nrow(y_true) == C)
    Ktrue <- ncol(y_true)
    Kuse  <- min(Ktrue, length(pnames))
    truth_long <- do.call(rbind, lapply(seq_len(C), function(ci) {
      data.frame(case = ci - 1L, param = pnames[seq_len(Kuse)], true = y_true[ci, seq_len(Kuse)])
    }))
    out <- merge(out, truth_long, by = c("case","param"), all.x = TRUE, sort = FALSE)
    out$bias <- out$mean - out$true
    out$covered_95 <- (out$true >= out$q025) & (out$true <= out$q975)
  }
  if (requireNamespace("tibble", quietly = TRUE)) tibble::as_tibble(out) else out
}

bf_compute_metrics <- function(sumr) {
  stopifnot(all(c("param","mean","true","q025","q975","covered_95","bias") %in% names(sumr)))
  sumr |>
    dplyr::group_by(param) |>
    dplyr::summarise(
      coverage_95 = mean(covered_95, na.rm = TRUE),
      mean_bias   = mean(bias, na.rm = TRUE),
      rmse        = sqrt(mean((mean - true)^2, na.rm = TRUE)),
      avg_width   = mean(q975 - q025, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::arrange(param)
}


# ===========
# 6) Plotting
# ===========
bf_plot_recovery <- function(sumr, param_order = NULL, y_stat = c("median","mean"),
                             pad_x = 0.005, pad_y = 0.005) {
  y_stat <- match.arg(y_stat)
  stopifnot(is.data.frame(sumr))
  stopifnot(requireNamespace("ggplot2"), requireNamespace("dplyr"), requireNamespace("patchwork"))
  library(dplyr); library(ggplot2)
  
  df <- sumr
  if (!is.null(param_order)) df$param <- factor(df$param, levels = param_order)
  yvar <- if (y_stat == "median") "median" else "mean"
  
  r2_tbl <- df %>%
    group_by(param) %>%
    summarise(r2 = { cc <- suppressWarnings(cor(true, .data[[yvar]], use="complete.obs")); if (is.na(cc)) NA_real_ else cc^2 },
              .groups = "drop")
  
  eps <- 1e-12
  pos_tbl <- df %>%
    group_by(param) %>%
    summarise(
      x_min = min(true, na.rm = TRUE), x_max = max(true, na.rm = TRUE),
      y_min = min(.data[[yvar]], na.rm = TRUE), y_max = max(.data[[yvar]], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      xpos = x_min + pad_x * (x_max - x_min + eps),
      ypos = y_max - pad_y * (y_max - y_min + eps)
    ) %>%
    left_join(r2_tbl, by = "param") %>%
    mutate(label = sprintf('R² = %.3f', r2))
  
  p_recov <- ggplot(df, aes(x = true, y = .data[[yvar]])) +
    geom_errorbar(aes(ymin = q025, ymax = q975), width = 0, color = "skyblue") +
    geom_point(alpha = 0.35, size = 1.5, color = "black") +
    geom_abline(slope = 1, intercept = 0, linewidth = 0.7, linetype = 2, color = "white") +
    geom_text(data = pos_tbl, aes(x = xpos, y = ypos, label = label), hjust = 0, vjust = 0.5, size = 3) +
    facet_wrap(~ param, scales = "free") +
    labs(x = "True value", y = if (y_stat == "median") "Posterior median" else "Posterior mean",
         title = "Parameter recovery (95% CI)") +
    theme_classic() +
    coord_cartesian(clip = "off") +
    scale_x_continuous(expand = expansion(mult = 0.02)) +
    scale_y_continuous(expand = expansion(mult = 0.02))
  
  met <- bf_compute_metrics(df)
  
  p_cov <- ggplot(met, aes(x = param, y = coverage_95)) +
    geom_col(color = "white", fill = "skyblue") +
    geom_hline(yintercept = 0.95, linetype = 2, color = "black") +
    coord_cartesian(ylim = c(0,1)) +
    labs(x = NULL, y = "95% CI coverage", title = "Coverage by parameter") +
    theme_classic()
  
  p_bias <- ggplot(df, aes(x = param, y = bias)) +
    geom_hline(yintercept = 0, linetype = 2) +
    geom_violin(trim = FALSE, alpha = 0.6) +
    geom_boxplot(width = 0.15, outlier.shape = NA) +
    labs(x = NULL, y = "Bias (mean - true)", title = "Bias distribution") +
    theme_classic()
  
  list(
    metrics = met,
    plots = list(recovery = p_recov, coverage = p_cov, bias = p_bias),
    patchwork = (p_recov / (p_cov | p_bias)) + patchwork::plot_layout(heights = c(2,1))
  )
}


# ==================
# 7) End-to-end run
# ==================
# Build simulator + workflow
bf_build_simulator_gaussian_set(
  n_trials = 400,
  mu_range = c(-1.0, 1.0),
  sigma_range = c(0.3, 1.5),
  standardize_x = FALSE,
  expose_split_keys = FALSE
)

bf_build_workflow_deepset(
  phi_units     = c(128,128),
  rho_units     = c(128,128),
  aggregation   = "mean",
  flow_type     = "coupling",
  n_flow_layers = 6,
  flow_hidden   = c(128,128),
  param_keys    = "theta",
  data_key      = "trials"
)

# Train
bf_fit_online(
  epochs = 5,
  num_batches_per_epoch = 100,
  batch_size = 200,
  with_validation = TRUE,
  n_val = 400
)

# Sample, summarise, plot
post  <- bf_sample_posteriors(n_test = 120, n_draws = 2000)
sumr  <- bf_summarise_draws(post, param_names = c("mu","sigma"))
out   <- bf_plot_recovery(sumr, param_order = c("mu","sigma"), y_stat = "median")

out$metrics     # coverage, bias, RMSE, width
out$patchwork   # recovery + coverage + bias composite
