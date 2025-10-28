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

# =========================
# 0) Utilities
# =========================
.stop_if_no_reticulate <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.")
  }
}

# =========================
# 1) Build simulator (Python)
# =========================
bf_build_simulator <- function(
    n_trials = 200,
    a_range  = c(0.5, 2.5),
    v_range  = c(0.5, 4.0),
    t0_range = c(0.15, 0.50)
) {
  .stop_if_no_reticulate()
  reticulate::py_run_string(sprintf("
import numpy as np, bayesflow as bf

# ---- config (globals in Python) ----
n_trials = %d
A_LOW, A_HIGH   = %f, %f
V_LOW, V_HIGH   = %f, %f
T0_LOW, T0_HIGH = %f, %f

def _simulate_racing_wald(n, a, v1, v2, t0, rng):
    mu1, mu2 = a / v1, a / v2
    lam = a * a
    t1 = t0 + rng.wald(mean=mu1, scale=lam, size=n)
    t2 = t0 + rng.wald(mean=mu2, scale=lam, size=n)
    rt  = np.minimum(t1, t2).astype(np.float32)
    win = (t2 < t1).astype(np.float32)
    return np.concatenate([rt, win], axis=0).astype(np.float32)

def prior():
    rng = np.random.default_rng()
    a  = rng.uniform(A_LOW,  A_HIGH)
    v1 = rng.uniform(V_LOW,  V_HIGH)
    v2 = rng.uniform(V_LOW,  V_HIGH)
    t0 = rng.uniform(T0_LOW, T0_HIGH)
    return dict(a=a, v1=v1, v2=v2, t0=t0,
                theta=np.array([a, v1, v2, t0], np.float32))

def likelihood(a, v1, v2, t0):
    x = _simulate_racing_wald(n_trials, float(a), float(v1), float(v2), float(t0),
                              np.random.default_rng())
    return dict(
        inference_conditions = x.astype(np.float32),                  # (2*n_trials,)
        inference_variables  = np.array([a, v1, v2, t0], np.float32)  # (4,)
    )

simulator = bf.make_simulator([prior, likelihood])
param_names = ['a','v1','v2','t0']
print('[ok] simulator ready: n_trials =', n_trials, '; input len =', 2*n_trials)
", as.integer(n_trials),
                                    a_range[1], a_range[2],
                                    v_range[1], v_range[2],
                                    t0_range[1], t0_range[2]))
  invisible(TRUE)
}


# Per-boundary summary features:
# x = [ accuracy, mean_rt, quantiles(rt | win==1)..., quantiles(rt | win==2)... ]
bf_build_simulator_summaries <- function(
    n_trials = 200,
    probs    = c(0.1, 0.3, 0.5, 0.7, 0.9),
    a_range  = c(0.5, 2.5),
    v_range  = c(0.5, 4.0),
    t0_range = c(0.15, 0.50)
) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.")
  }
  probs_py <- paste0("[", paste(sprintf("%.10f", probs), collapse = ","), "]")
  
  reticulate::py_run_string(sprintf("
import numpy as np, bayesflow as bf

# ---------- config ----------
n_trials = %d
A_LOW, A_HIGH   = %f, %f
V_LOW, V_HIGH   = %f, %f
T0_LOW, T0_HIGH = %f, %f
Q_PROBS = np.array(%s, dtype=float)   # quantile probs, e.g., [0.1,0.3,0.5,0.7,0.9]

def _simulate_trials(n, a, v1, v2, t0, rng):
    # Wald draws for each accumulator; decision is the min
    mu1, mu2 = a / v1, a / v2
    lam = a * a
    t1 = t0 + rng.wald(mean=mu1, scale=lam, size=n)
    t2 = t0 + rng.wald(mean=mu2, scale=lam, size=n)
    rt   = np.minimum(t1, t2).astype(np.float32)          # [n]
    win1 = (t1 < t2)                                      # True if acc1 faster
    resp = win1.astype(np.float32)                        # 1 for acc1, 0 for acc2
    return rt, resp, win1

def _safe_quantiles(x, probs, fallback):
    if x.size == 0:
        return np.quantile(fallback, probs).astype(np.float32)
    return np.quantile(x, probs).astype(np.float32)

def prior():
    rng = np.random.default_rng()
    a  = rng.uniform(A_LOW,  A_HIGH)
    v1 = rng.uniform(V_LOW,  V_HIGH)
    v2 = rng.uniform(V_LOW,  V_HIGH)
    t0 = rng.uniform(T0_LOW, T0_HIGH)
    theta = np.array([a, v1, v2, t0], dtype=np.float32)
    return dict(a=a, v1=v1, v2=v2, t0=t0, theta=theta)

def likelihood(a, v1, v2, t0):
    rng = np.random.default_rng()
    rt, resp, win1 = _simulate_trials(n_trials, float(a), float(v1), float(v2), float(t0), rng)
    acc = float(resp.mean())                               # P(win==1)
    mu  = float(rt.mean())                                 # overall mean RT

    # Per-boundary RT quantiles (conditional)
    rt1 = rt[win1]                                         # RTs where acc1 won
    rt2 = rt[~win1]                                        # RTs where acc2 won
    q1  = _safe_quantiles(rt1, Q_PROBS, fallback=rt)       # [len(probs)]
    q2  = _safe_quantiles(rt2, Q_PROBS, fallback=rt)       # [len(probs)]

    # Feature vector: [accuracy, mean_rt, q1..., q2...]
    x = np.concatenate(([acc, mu], q1, q2)).astype(np.float32)

    return dict(
        inference_conditions = x,                                  # (2 + 2*len(probs),)
        inference_variables  = np.array([a, v1, v2, t0], np.float32)  # (4,)
    )

simulator = bf.make_simulator([prior, likelihood])
param_names = ['a','v1','v2','t0']
print('[ok] per-boundary summary simulator:',
      'n_trials =', n_trials,
      '; features =', 2 + 2*len(Q_PROBS),
      '(acc, mean, q(rt|win=1), q(rt|win=2))')
", as.integer(n_trials),
                                    a_range[1], a_range[2],
                                    v_range[1], v_range[2],
                                    t0_range[1], t0_range[2],
                                    probs_py))
  invisible(TRUE)
}


bf_build_simulator_rawtrials_set <- function(
    n_trials          = 400,
    a_range           = c(1.0, 1.5),
    v1_range          = c(1.0, 2.0),   # <-- new: separate range for v1
    v2_range          = c(1.0, 2.0),   # <-- new: separate range for v2
    t0_range          = c(0.2, 0.3),
    standardize_rt    = FALSE,         # per-case z-score RTs if TRUE
    expose_split_keys = FALSE          # also expose a,v1,v2,t0 keys if TRUE
) {
  stopifnot(requireNamespace("reticulate", quietly = TRUE))
  stopifnot(length(a_range)  == 2L,
            length(v1_range) == 2L,
            length(v2_range) == 2L,
            length(t0_range) == 2L)
  
  T  <- as.integer(n_trials)
  ex <- if (expose_split_keys) "True" else "False"
  
  reticulate::py_run_string(sprintf("
import numpy as np

# ---------------- Config from R ----------------
_T        = %d
a_min, a_max     = %f, %f
v1_min, v1_max   = %f, %f
v2_min, v2_max   = %f, %f
t0_min, t0_max   = %f, %f
standardize_rt   = %s
expose_split     = %s

# ---------------- Utilities ----------------
def _as_int(n):
    \"\"\"Coerce an int-like value; accept tuples/lists like (B,) or [B].\"\"\"
    try:
        return int(n)
    except (TypeError, ValueError):
        try:
            return int(n[0])
        except Exception as e:
            raise TypeError(f'Cannot coerce n={n} to int') from e

# ---------------- Racing Wald helpers ----------------
def _wald_times(a, v, size):
    # Inverse Gaussian (Wald): mean=mu=a/v, shape=lambda=a^2
    v  = np.maximum(v, 1e-6)
    mu = a / v
    lam = (a * a)
    return np.random.wald(mean=mu, scale=lam, size=size)

def _simulate_case(theta_row):
    a, v1, v2, t0 = theta_row
    t1 = _wald_times(a, v1, _T)
    t2 = _wald_times(a, v2, _T)
    rt = t0 + np.minimum(t1, t2)
    resp = (t1 <= t2).astype(np.float32)  # 1 if acc1 wins else 0

    if standardize_rt:
        m, s = rt.mean(), rt.std()
        if s > 1e-9:
            rt = (rt - m) / s

    trials = np.stack([rt.astype(np.float32), resp], axis=1)  # [T, 2]
    return trials

def _sample_params(n):
    a  = np.random.uniform(a_min,  a_max,  size=n).astype(np.float32)
    v1 = np.random.uniform(v1_min, v1_max, size=n).astype(np.float32)
    v2 = np.random.uniform(v2_min, v2_max, size=n).astype(np.float32)
    t0 = np.random.uniform(t0_min, t0_max, size=n).astype(np.float32)
    theta = np.stack([a, v1, v2, t0], axis=1)  # [n, 4]
    return theta

# ---------------- Simulator classes ----------------
class RawTrialsSimulator:
    def __init__(self, T):
        self.T = T
    def sample(self, n):
        n = _as_int(n)
        theta = _sample_params(n)                      # [n, 4]
        X = np.empty((n, self.T, 2), dtype=np.float32)
        for i in range(n):
            X[i] = _simulate_case(theta[i])
        return {'theta': theta.astype(np.float32), 'trials': X}

class KeySplitSimulator:
    \"\"\"Adds a,v1,v2,t0 keys alongside theta and trials.\"\"\"
    def __init__(self, base):
        self.base = base
        self.T = getattr(base, 'T', None)
    def sample(self, n):
        d = self.base.sample(n)
        th = d['theta']
        d['a']  = th[:, 0].astype('float32')
        d['v1'] = th[:, 1].astype('float32')
        d['v2'] = th[:, 2].astype('float32')
        d['t0'] = th[:, 3].astype('float32')
        return d

# ---------------- Expose `simulator` ----------------
simulator = RawTrialsSimulator(_T)
if expose_split:
    simulator = KeySplitSimulator(simulator)

# Smoke test
_test = simulator.sample(2)
print('[sim ok] keys:', list(_test.keys()), '| theta', _test['theta'].shape, '| trials', _test['trials'].shape)
", T,
                                    a_range[1], a_range[2],
                                    v1_range[1], v1_range[2],
                                    v2_range[1], v2_range[2],
                                    t0_range[1], t0_range[2],
                                    if (standardize_rt) "True" else "False",
                                    ex
  ))
  
  invisible(TRUE)
}




# More flexible network ---------------------------------------------------

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
  if (with_validation) {
    reticulate::py_run_string(sprintf("val_raw = _sim_obj.sample(%d)", as.integer(N_val)))
    val_arg <- "validation_data=val_raw"
  } else {
    val_arg <- ""
  }
  reticulate::py_run_string(sprintf(
    "history = workflow.fit_online(epochs=%d, batch_size=%d, num_batches_per_epoch=%d%s%s)",
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

# Summarise either a 3-D array OR the posterior df directly.
bf_summarise_draws <- function(x, Y_true = NULL, param_names = NULL) {
  if (is.data.frame(x)) {
    df <- x
    A  <- bf_as_draws_array(df)
    if (is.null(Y_true)) Y_true <- bf_extract_truth(df)
  } else {
    A <- x
  }
  stopifnot(length(dim(A)) == 3L)
  D <- dim(A)[1]; C <- dim(A)[2]; K <- dim(A)[3]
  
  # Names
  pnames <- dimnames(A)[[3]]
  if (!is.null(param_names)) {
    stopifnot(length(param_names) == K)
    pnames <- param_names
  }
  
  # Compute summaries
  mean_mat   <- apply(A, c(2,3), mean,   na.rm = TRUE)   # [C,K]
  median_mat <- apply(A, c(2,3), median, na.rm = TRUE)
  q025_mat   <- apply(A, c(2,3), quantile, probs = 0.025, na.rm = TRUE)
  q975_mat   <- apply(A, c(2,3), quantile, probs = 0.975, na.rm = TRUE)
  sd_mat     <- apply(A, c(2,3), sd,      na.rm = TRUE)
  
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
  
  # Attach truth if provided
  if (!is.null(Y_true)) {
    if (is.vector(Y_true)) {
      Y_true <- matrix(rep(Y_true, each = C), nrow = C, byrow = TRUE)
    }
    stopifnot(nrow(Y_true) == C)
    Ktrue <- ncol(Y_true)
    if (Ktrue >= K) {
      truth_long <- do.call(rbind, lapply(seq_len(C), function(ci) {
        data.frame(case = ci - 1L, param = pnames, true = Y_true[ci, seq_len(K)])
      }))
      out <- merge(out, truth_long, by = c("case","param"), all.x = TRUE, sort = FALSE)
      out$bias <- out$mean - out$true
      out$covered_95 <- (out$true >= out$q025) & (out$true <= out$q975)
    }
  }
  if (requireNamespace("tibble", quietly = TRUE)) out <- tibble::as_tibble(out)
  out
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
    geom_abline(slope = 1, intercept = 0, linewidth = 0.8, linetype = 2, color = "black") +
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

# Optional: pull ground truth from the posterior df if present (true_p*)
bf_extract_truth <- function(df) {
  tcols <- grep("^true_p\\d+$", names(df), value = TRUE)
  if (!length(tcols)) return(NULL)
  C <- length(unique(df$case))
  # keep one row per case (first draw block), then arrange by case
  one <- df[df$draw == min(df$draw), c("case", tcols), drop = FALSE]
  one <- one[order(one$case), , drop = FALSE]
  as.matrix(one[, tcols, drop = FALSE])   # [C, K_true]
}

# ---- Metrics from sumr tibble ----
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

# -------------------------------------------------------------------------

# 1) Define the simulator (creates Python `simulator` with robust .sample)
bf_build_simulator_rawtrials_set(
  n_trials = 400,
  a_range  = c(1.0, 1.5),
  v1_range = c(1.2, 2.2), 
  v2_range = c(0.8, 1.6),
  t0_range = c(0.2, 0.3),
  standardize_rt = TRUE,
  expose_split_keys = FALSE
)

# 2) Rebuild your workflow (use param_keys='theta' or c('a','v1','v2','t0') if expose_split_keys=TRUE)
bf_build_workflow_deepset(
  phi_units     = c(256,256,128),
  rho_units     = c(256,128),
  aggregation   = "mean",
  flow_type     = "coupling",
  n_flow_layers = 8,
  flow_hidden   = c(256,256,256),
  param_keys    = "theta",    # or c('a','v1','v2','t0') if expose_split_keys=TRUE
  data_key      = "trials"
)


# 3) Train (tuple-safe now)
bf_fit_online(epochs = 10, num_batches_per_epoch = 100, batch_size = 256,
              with_validation = TRUE, N_val = 512)


# 4) Sample & plot recovery
res  <- bf_sample_posteriors(N_test = 150, num_samples = 3000)
sumr <- bf_summarise_draws(res, param_names = c("a","v1","v2","t0"))
out  <- bf_plot_recovery(sumr, param_order = c("a","v1","v2","t0"), y_stat = "median")
out$metrics         # coverage, mean bias, RMSE, avg width
out$patchwork       # combined figure
# Or individual:
out$plots$recovery
out$plots$coverage
out$plots$bias
