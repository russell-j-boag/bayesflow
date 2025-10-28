rm(list = ls())

# --- STEP 1: environment / versions -------------------------------------
options(reticulate.use_uv = FALSE)
Sys.setenv(RETICULATE_PYTHON = "/Users/rjb779/Library/r-miniconda-arm64/envs/r-bf/bin/python")
Sys.setenv(KERAS_BACKEND = "jax")

library("reticulate")
py_config()

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


# -------------------------------------------------------------------------

# Utilities
.stop_if_no_reticulate <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.")
  }
}

# Amortise over T by randomising the set size per call, and allow pinning T.
bf_set_T   <- function(T) reticulate::py_run_string(sprintf("_bf_set_T(%d)", as.integer(T)))
bf_clear_T <- function()  reticulate::py_run_string("_bf_clear_T()")

bf_build_simulator_gaussian_set <- function(
    min_trials        = 20L,
    max_trials        = 400L,
    mu_range          = c(-1.0, 1.0),
    sigma_range       = c(0.3, 1.5),
    standardize_x     = FALSE,
    expose_split_keys = FALSE
) {
  stopifnot(requireNamespace("reticulate", quietly = TRUE))
  stopifnot(length(mu_range) == 2L, length(sigma_range) == 2L)
  stopifnot(min_trials >= 1L, max_trials >= min_trials)
  ex <- if (expose_split_keys) "True" else "False"
  
  reticulate::py_run_string(sprintf("
import numpy as np

# ------------ Config from R ------------
T_min, T_max = int(%d), int(%d)
mu_min, mu_max = %f, %f
s_min,  s_max  = %f, %f
standardize_x  = %s
expose_split   = %s

def _sample_params(n):
    mu    = np.random.uniform(mu_min, mu_max, size=n).astype(np.float32)
    sigma = np.random.uniform(s_min,  s_max,  size=n).astype(np.float32)
    return np.stack([mu, sigma], axis=1)  # [n, 2]

def _simulate_case(theta_row, T):
    mu, sigma = float(theta_row[0]), float(theta_row[1])
    x = np.random.normal(loc=mu, scale=max(sigma, 1e-6), size=T).astype(np.float32)
    if standardize_x:
        m, s = x.mean(), x.std()
        if s > 1e-9: x = (x - m) / s
    return x[:, None]  # [T, 1]

class GaussianSetSimulator:
    def __init__(self, T_min, T_max):
        self.T_min, self.T_max = int(T_min), int(T_max)
        self._fixed_T = None  # if set, use this T

    def set_fixed_T(self, T=None):
        if T is None:
            self._fixed_T = None
        else:
            T = int(T)
            if not (self.T_min <= T <= self.T_max):
                raise ValueError(f'Fixed T={T} outside [{self.T_min}, {self.T_max}]')
            self._fixed_T = T

    def sample(self, n):
        n = int(n)
        T = self._fixed_T if self._fixed_T is not None else np.random.randint(self.T_min, self.T_max + 1)
        theta = _sample_params(n).astype(np.float32)         # [n, 2]
        X = np.empty((n, T, 1), dtype=np.float32)
        for i in range(n):
            X[i] = _simulate_case(theta[i], T)
        # also return T per case so R can annotate
        return {'theta': theta, 'trials': X, 'T': np.full((n,), T, dtype=np.int32)}

class KeySplitSimulator:
    def __init__(self, base): self.base = base
    def set_fixed_T(self, T=None): self.base.set_fixed_T(T)
    def sample(self, n):
        d  = self.base.sample(n)
        th = d['theta']
        d['mu']    = th[:, 0].astype('float32')
        d['sigma'] = th[:, 1].astype('float32')
        return d

# ------------ Expose `simulator` ------------
_base = GaussianSetSimulator(T_min, T_max)
simulator = KeySplitSimulator(_base) if expose_split else _base

# Convenience helpers for R
def _bf_set_T(T): simulator.set_fixed_T(T)
def _bf_clear_T(): simulator.set_fixed_T(None)

# Smoke test
_test = simulator.sample(3)
print('[sim ok] keys:', list(_test.keys()), '| theta', _test['theta'].shape, '| trials', _test['trials'].shape, '| T', _test['T'][0])
", as.integer(min_trials), as.integer(max_trials),
                                    mu_range[1], mu_range[2], sigma_range[1], sigma_range[2],
                                    if (standardize_x) "True" else "False", ex))
  invisible(TRUE)
}


bf_build_workflow_deepset <- function(
    # DeepSet
  phi_units     = c(256, 256, 128),
  rho_units     = c(256, 128),
  aggregation   = c("mean","sum","max"),
  # Flow
  flow_type     = c("coupling", "autoregressive"),
  n_flow_layers = 8L,
  flow_hidden   = c(256, 256),
  # Data keys
  param_keys    = "theta",
  data_key      = "trials"
) {
  stopifnot(requireNamespace("reticulate", quietly = TRUE))
  aggregation <- match.arg(aggregation)
  flow_type   <- match.arg(flow_type)
  
  phi_py   <- paste0("[", paste(as.integer(phi_units), collapse=","), "]")
  rho_py   <- paste0("[", paste(as.integer(rho_units), collapse=","), "]")
  flowhid  <- paste0("[", paste(as.integer(flow_hidden), collapse=","), "]")
  
  param_keys_vec <- as.character(param_keys)
  py_param_list  <- paste0("[", paste(sprintf("'%s'", param_keys_vec), collapse = ", "), "]")
  use_concat     <- length(param_keys_vec) > 1L
  data_key_py    <- sprintf("'%s'", as.character(data_key))
  
  # Ensure a Python simulator exists and expose .sample(n)
  reticulate::py_run_string(paste(
    "import builtins",
    "if 'simulator' not in globals() and not hasattr(builtins, 'simulator'):",
    "    raise RuntimeError('Python `simulator` is not defined.')",
    "",
    "class CallableSimulator:",
    "    def __init__(self, fn): self.fn = fn",
    "    def sample(self, n):    return self.fn(n)",
    "",
    "if hasattr(simulator, 'sample') and callable(getattr(simulator, 'sample')):",
    "    _sim_obj = simulator",
    "elif callable(simulator):",
    "    _sim_obj = CallableSimulator(simulator)",
    "else:",
    "    raise TypeError('`simulator` must be callable or have a `.sample(n)` method.')",
    sep = "\n"
  ))
  
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
  
  py_code <- paste(
    "import numpy as np",
    "import bayesflow as bf",
    "",
    "# ---- Adapter ----",
    sprintf("adapter = %s", adapter_block),
    "",
    "# ---- Summary network (DeepSet) ----",
    sprintf(
      "summary_net = bf.networks.DeepSet(phi_units=%s, rho_units=%s, aggregation='%s')",
      phi_py, rho_py, aggregation
    ),
    "",
    "# ---- Inference network (Flow) ----",
    "try:",
    if (flow_type == "coupling") {
      sprintf(
        "    inference_net = bf.networks.CouplingFlow(n_coupling_layers=%d, hidden_units=%s)",
        as.integer(n_flow_layers), flowhid
      )
    } else {
      sprintf(
        "    inference_net = bf.networks.AutoregressiveFlow(n_layers=%d, hidden_units=%s)",
        as.integer(n_flow_layers), flowhid
      )
    },
    "except TypeError as e:",
    "    print('[warn] Flow kwargs not supported in this BF version; using defaults:', e)",
    if (flow_type == "coupling") {
      "    inference_net = bf.networks.CouplingFlow()"
    } else {
      "    inference_net = bf.networks.AutoregressiveFlow()"
    },
    "",
    "# ---- Workflow ----",
    "workflow = bf.BasicWorkflow(",
    "    simulator = _sim_obj,",
    "    adapter   = adapter,",
    "    inference_network  = inference_net,",
    "    summary_network    = summary_net,",
    "    inference_variables = 'inference_variables',",
    "    summary_variables   = 'summary_variables'",
    ")",
    "",
    "# ---- Sanity check ----",
    "_raw = _sim_obj.sample(4)",
    "_bat = adapter(_raw)",
    "shapes = {k: (np.array(v).shape if hasattr(v, '__array__') else getattr(v, 'shape', None))",
    "          for k, v in _bat.items()}",
    "print('[ok] v2 workflow ready; adapted shapes:', shapes)",
    sep = "\n"
  )
  
  reticulate::py_run_string(py_code)
  invisible(TRUE)
}


bf_fit_online <- function(epochs = 10,
                          num_batches_per_epoch = 200,
                          batch_size = 256,
                          with_validation = TRUE,
                          N_val = 512) {
  stopifnot(requireNamespace("reticulate", quietly = TRUE))
  
  # -- Prepare validation data (if requested)
  if (with_validation) {
    py_run_string(sprintf("
# --- draw validation cases with variable T ---
import numpy as np
val_raw = _sim_obj.sample(%d)
if 'trials' in val_raw:
    data_key = 'trials'
elif 'x' in val_raw:
    data_key = 'x'
else:
    raise KeyError('Could not find a data key (expected \"trials\" or \"x\").')
N = val_raw[data_key].shape[0]
Tvec = val_raw.get('T', np.full((N,), val_raw[data_key].shape[1], dtype=np.int32))
print(f'[val] validation_data built with N={N}, T∈[{Tvec.min()}, {Tvec.max()}]')
", as.integer(N_val)))
    val_arg <- "validation_data=val_raw"
  } else {
    val_arg <- ""
  }
  
  # -- Patch for tuple-safe n argument (BayesFlow sometimes passes (batch_size,))
  py_run_string("
import numpy as np
def _bf_as_int(n):
    try:
        return int(n)
    except Exception:
        try:
            return int(n[0])
        except Exception as e:
            raise TypeError(f'Cannot coerce n={n!r} to int') from e

if 'GaussianSetSimulator' in globals():
    def _sample_safe(self, n):
        n = _bf_as_int(n)
        T = getattr(self, '_fixed_T', None)
        if T is None:
            T = np.random.randint(self.T_min, self.T_max + 1)
        theta = _sample_params(n).astype(np.float32)
        X = np.empty((n, T, 1), dtype=np.float32)
        for i in range(n):
            X[i] = _simulate_case(theta[i], T)
        return {'theta': theta, 'trials': X, 'T': np.full((n,), T, dtype=np.int32)}
    GaussianSetSimulator.sample = _sample_safe
print('[fit] Patched GaussianSetSimulator.sample to handle tuple n.')
")
  
  # -- Train
  py_run_string(sprintf("
print('[fit] Starting training for %d epochs × %d batches, batch_size=%d')
history = workflow.fit_online(
    epochs=%d,
    batch_size=%d,
    num_batches_per_epoch=%d%s%s
)
print('[fit ok] Training completed.')
", 
                as.integer(epochs), as.integer(num_batches_per_epoch), as.integer(batch_size),
                as.integer(epochs), as.integer(batch_size), as.integer(num_batches_per_epoch),
                if (with_validation) ", " else "",
                val_arg
  ))

invisible(TRUE)
}


bf_sample_posteriors <- function(N_test = 200, num_samples = 2000) {
  stopifnot(requireNamespace("reticulate", quietly = TRUE))
  # Python: sample, reshape, and build a tidy table
  reticulate::py_run_string(sprintf("
import numpy as np
import pandas as pd

# 1) Draw raw test sets
_raw = _sim_obj.sample(%d)

# 2) Detect the raw data key used BEFORE the adapter
if 'trials' in _raw:
    data_key = 'trials'
elif 'x' in _raw:
    data_key = 'x'
else:
    raise KeyError('Could not find a data key (expected \"trials\" or \"x\").')

X = _raw[data_key]   # shape [N_test, T, D]
N = X.shape[0]

# 3) Sample posteriors
_samples = workflow.sample(num_samples=%d, conditions={data_key: X})

# 4) Extract the posterior tensor (handle dict/tuple variations)
S = None
if isinstance(_samples, dict):
    # common key names to try
    for k in ['inference_variables', 'posterior', 'theta', 'z', 'y']:
        if k in _samples:
            S = _samples[k]
            break
    if S is None:
        # if dict-of-dicts (e.g., by parameter blocks), stack last axis
        # S will be a list of arrays to concatenate along last dim
        parts = []
        keys = []
        for k, v in _samples.items():
            if isinstance(v, np.ndarray):
                parts.append(v)
                keys.append(k)
        if parts:
            # ensure same shape except last dim
            S = np.concatenate(parts, axis=-1)
        else:
            raise ValueError('Unrecognized structure in samples dict: cannot find posterior array.')
elif isinstance(_samples, (list, tuple)):
    # take the first array-like entry
    for v in _samples:
        if isinstance(v, np.ndarray):
            S = v
            break
else:
    # maybe it's already an ndarray
    if isinstance(_samples, np.ndarray):
        S = _samples

if S is None:
    raise ValueError('Could not locate posterior samples array in the return of workflow.sample(...)')

# 5) Normalize axis order to (draw, case, param)
shape = S.shape
if S.ndim != 3:
    raise ValueError(f'Expected posterior array with 3 dims (draw, case, param), got shape {shape}')

DRAWS, CASES, DPAR = None, None, None

# Identify which axis is draws by matching num_samples
axes = list(range(3))
if %d in shape:
    draw_axis = shape.index(%d)
else:
    # fallback: assume axis 0 is draws
    draw_axis = 0

# Identify which axis is cases by matching N
if N in shape:
    case_axis = shape.index(N)
else:
    # fallback: assume a different axis is cases
    candidates = [ax for ax in axes if ax != draw_axis]
    case_axis = candidates[0]

# The remaining axis is params
param_axis = [ax for ax in axes if ax not in (draw_axis, case_axis)][0]

# Permute to (draw, case, param)
if (draw_axis, case_axis, param_axis) != (0,1,2):
    S = np.transpose(S, (draw_axis, case_axis, param_axis))

DRAWS, CASES, DPAR = S.shape  # should be (num_samples, N_test, D_theta)

# 6) Build a flat table: columns = p0..p{DPAR-1}, plus draw, case
flat = S.reshape(DRAWS * CASES, DPAR)
df = pd.DataFrame(flat, columns=[f'p{j}' for j in range(DPAR)])
df['draw'] = np.repeat(np.arange(DRAWS), CASES)
df['case'] = np.tile(np.arange(CASES), DRAWS)

# 7) Optionally append true theta (if available) aligned by case
theta_true = None
for k_true in ['theta', 'inference_variables', 'params', 'prior']:
    if k_true in _raw:
        theta_true = _raw[k_true]
        break

if isinstance(theta_true, np.ndarray):
    # theta_true: [N, DPAR_true]; repeat over draws to align
    if theta_true.ndim == 1:
        theta_true = theta_true.reshape(N, 1)
    DPAR_true = theta_true.shape[1]
    th_flat = np.repeat(theta_true, DRAWS, axis=0)  # [N*DRAWS, DPAR_true] but order is by case fastest or draw fastest?
    # Our df ordering is draw-major then case; th_flat is case-major repeated by draws → need to interleave:
    th_flat = np.vstack([theta_true for _ in range(DRAWS)])  # stack draws blocks
    for j in range(DPAR_true):
        df[f'true_p{j}'] = th_flat[:, j]

# 8) Stash to a global for R to pull
bf_last_samples_df = df
print('[ok] posterior table:', bf_last_samples_df.shape, ' (columns:', list(bf_last_samples_df.columns)[:8], '...)')
", as.integer(N_test), as.integer(num_samples), as.integer(num_samples), as.integer(num_samples)))
  
  # Bring the pandas DataFrame back to R as a tibble
  out <- reticulate::py$bf_last_samples_df
  # Convert to R tibble if available
  if (requireNamespace("tibble", quietly = TRUE)) out <- tibble::as_tibble(out)
  out
}


# --- Run across a grid of T values -----------------------------------
bf_sample_posteriors_by_T <- function(
    T_values,
    cases_per_T = 40L,
    num_samples = 2000L,
    sampler = NULL
) {
  if (is.null(sampler)) {
    if (!exists("bf_sample_posteriors", mode = "function"))
      stop("Provide a `sampler` function or define `bf_sample_posteriors()` first.")
    sampler <- bf_sample_posteriors
  }
  
  out <- lapply(as.integer(T_values), function(Ti) {
    bf_set_T(Ti)
    df <- sampler(N_test = as.integer(cases_per_T),
                  num_samples = as.integer(num_samples))
    # Ensure required columns exist
    if (!"T" %in% names(df))   df$T <- as.integer(Ti)
    if (!"draw" %in% names(df)) stop("Sampler must return a 'draw' column.")
    if (!"case" %in% names(df)) stop("Sampler must return a 'case' column.")
    df
  })
  bf_clear_T()
  dplyr::bind_rows(out)
}


# Helper to compute summaries per T
bf_summarise_by_T <- function(df_post, param_names = c("mu","sigma")) {
  stopifnot(all(c("T","draw","case") %in% names(df_post)))
  split_dfs <- split(df_post, df_post$T)
  sums <- lapply(names(split_dfs), function(k) {
    dT <- split_dfs[[k]]
    s  <- bf_summarise_draws(dT, param_names = param_names)
    s$T <- as.integer(k)
    s
  })
  dplyr::bind_rows(sums)
}


bf_plot_recovery <- function(sumr, param_order = NULL, y_stat = c("median","mean"),
                             pad_x = 0.005, pad_y = 0.005) {
  stopifnot(is.data.frame(sumr))
  y_stat <- match.arg(y_stat)
  
  if (!requireNamespace("ggplot2", quietly = TRUE) ||
      !requireNamespace("dplyr", quietly = TRUE) ||
      !requireNamespace("patchwork", quietly = TRUE)) {
    stop("Please install ggplot2, dplyr, patchwork.")
  }
  library(dplyr); library(ggplot2)
  
  df <- sumr
  if (!is.null(param_order)) df$param <- factor(df$param, levels = param_order)
  yvar <- if (y_stat == "median") "median" else "mean"
  
  # R²: true vs posterior median (as requested)
  r2_tbl <- df %>%
    group_by(param) %>%
    summarise(
      r2 = { cc <- suppressWarnings(cor(true, median, use = "complete.obs")); if (is.na(cc)) NA_real_ else cc^2 },
      .groups = "drop"
    )
  
  # Corner positions with *tight* padding
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
    mutate(label = sprintf("R² = %.3f", r2))
  
  p_recov <- ggplot(df, aes(x = true, y = .data[[yvar]])) +
    geom_errorbar(aes(ymin = q025, ymax = q975), width = 0, color = "skyblue") +
    geom_point(alpha = 0.35, size = 1.5, color = "black") +
    geom_abline(slope = 1, intercept = 0, linewidth = 0.7, linetype = 2, color = "white") +
    geom_text(
      data = pos_tbl,
      aes(x = xpos, y = ypos, label = label),
      hjust = 0, vjust = 0.5, size = 3
    ) +
    facet_wrap(~ param, scales = "free") +
    labs(
      x = "True value",
      y = if (y_stat == "median") "Posterior median" else "Posterior mean",
      title = "Parameter recovery (95% CI)"
    ) +
    theme_classic() +
    coord_cartesian(clip = "off") +                     # allows ultra-tight placement
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


# Convert the posterior tibble/data.frame returned by bf_sample_posteriors()
# into an array with shape [n_draws, n_cases, n_params].
bf_as_draws_array <- function(df) {
  stopifnot(is.data.frame(df))
  stopifnot(all(c("draw","case") %in% names(df)))
  pcols <- grep("^p\\d+$", names(df), value = TRUE)
  if (length(pcols) == 0) stop("No columns matching ^p\\d+$ found.")
  
  df <- df[order(df$draw, df$case), c("draw","case", pcols), drop = FALSE]
  M  <- as.matrix(df[, pcols, drop = FALSE])                # (n_draws*n_cases) x K
  D  <- length(unique(df$draw))
  C  <- length(unique(df$case))
  K  <- length(pcols)
  
  arr <- array(NA_real_, dim = c(D, C, K))
  for (j in seq_len(K)) {
    # Rows are grouped by draw, within each draw cases increase → fill row-wise
    arr[,,j] <- matrix(M[, j], nrow = D, ncol = C, byrow = TRUE)
  }
  dimnames(arr) <- list(draw = NULL, case = seq_len(C) - 1L, param = pcols)
  arr
}

# Pull ground truth from the posterior df if present (true_p*)
bf_extract_truth <- function(df) {
  tcols <- grep("^true_p\\d+$", names(df), value = TRUE)
  if (!length(tcols)) return(NULL)
  C <- length(unique(df$case))
  # keep one row per case (first draw block), then arrange by case
  one <- df[df$draw == min(df$draw), c("case", tcols), drop = FALSE]
  one <- one[order(one$case), , drop = FALSE]
  as.matrix(one[, tcols, drop = FALSE])   # [C, K_true]
}

# Metrics from sumr tibble ----
bf_compute_metrics <- function(sumr) {
  stopifnot(all(c("param","mean","true","q025","q975","covered_95","bias") %in% names(sumr)))
  met <- sumr |>
    dplyr::group_by(param) |>
    dplyr::summarise(
      coverage_95 = mean(covered_95, na.rm = TRUE),
      mean_bias   = mean(bias, na.rm = TRUE),
      rmse        = sqrt(mean((mean - true)^2, na.rm = TRUE)),
      avg_width   = mean(q975 - q025, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::arrange(param)
  met
}

save_recovery_pages_by_T <- function(sumr_T,
                                     y_stat   = "median",
                                     out_dir  = "plots",
                                     filename = "recovery_by_T.pdf") {
  stopifnot(requireNamespace("dplyr", quietly = TRUE))
  stopifnot(requireNamespace("patchwork", quietly = TRUE))
  
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  Ts <- sort(unique(sumr_T$T))
  pdf_path <- file.path(out_dir, filename)
  
  grDevices::pdf(pdf_path, width = 10, height = 8)
  on.exit(grDevices::dev.off(), add = TRUE)
  
  for (Ti in Ts) {
    out <- bf_plot_recovery(
      sumr_T %>% dplyr::filter(T == Ti),
      param_order = c("mu", "sigma"),
      y_stat = y_stat
    )
    p <- out$patchwork + patchwork::plot_annotation(
      title = sprintf("Parameter recovery at T = %d", Ti)
    )
    print(p)  # one page per T
  }
  message(sprintf("[saved] %s with %d pages.", pdf_path, length(Ts)))
  invisible(pdf_path)
}


# -------------------------------------------------------------------------

# Build simulator over a T range
bf_build_simulator_gaussian_set(
  min_trials        = 100,
  max_trials        = 1000,
  mu_range          = c(-1.0, 1.0),
  sigma_range       = c(0.3, 1.5),
  standardize_x     = FALSE,
  expose_split_keys = FALSE
)


# Build workflow
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


# Train — each batch uses a fresh T sampled uniformly from [20, 400]
bf_fit_online(epochs = 5, num_batches_per_epoch = 100, batch_size = 256,
              with_validation = TRUE, N_val = 400)

# Example end-to-end evaluation over multiple T values
# (after training with randomised T)
df_post_T <- bf_sample_posteriors_by_T(
  T_values    = c(100L, 200L, 500L, 1000L),
  cases_per_T = 100L,
  num_samples = 2000L
)

# Sanity check:
print(names(df_post_T))
print(table(df_post_T$T))

sumr_T <- bf_summarise_by_T(df_post_T, param_names = c("mu","sigma"))


save_recovery_pages_by_T(sumr_T, y_stat = "median", out_dir = "plots",
                         filename = "recovery_by_T.pdf")

