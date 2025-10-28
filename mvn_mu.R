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

# --- NEW: MVN simulator ---------------------------------------------------
# --- MVN (mu, Sigma) simulator using np.random.multivariate_normal ----------
bf_build_simulator_mvn_full_set <- function(
    min_trials        = 100L,
    max_trials        = 2000L,
    D                 = 3L,                    # data dimensionality
    mu_range          = c(-1.0, 1.0),         # per-dim Uniform for mu
    L_diag_range      = c(0.3, 2.0),          # positive diag entries for L
    L_offdiag_sd      = 0.3,                  # off-diagonal ~ N(0, sd)
    standardize_x     = FALSE,                # typically FALSE for MVN
    expose_split_keys = TRUE                  # also expose 'mu' and 'L_packed'
) {
  stopifnot(D >= 1L, length(mu_range) == 2L, length(L_diag_range) == 2L)
  
  ex <- if (expose_split_keys) "True" else "False"
  
  reticulate::py_run_string(sprintf("
import numpy as np

# --------------------- Config from R ---------------------
T_min, T_max = int(%d), int(%d)
D            = int(%d)
mu_min, mu_max     = %f, %f
Ldiag_min, Ldiag_max = %f, %f
L_offdiag_sd       = %f
standardize_x      = %s
expose_split       = %s

K = D + (D*(D+1))//2  # total theta length: D mus + vech(L)

def _sample_mu(n):
    # mu ~ U(mu_min, mu_max)^D
    return np.random.uniform(mu_min, mu_max, size=(n, D)).astype(np.float32)

def _sample_L(n):
    # Sample lower-triangular L with positive diagonals
    Ls = np.zeros((n, D, D), dtype=np.float32)
    # Diagonals: Uniform(Ldiag_min, Ldiag_max)
    diag = np.random.uniform(Ldiag_min, Ldiag_max, size=(n, D)).astype(np.float32)
    for i in range(D):
        Ls[:, i, i] = diag[:, i]
    # Strict lower triangle: Normal(0, L_offdiag_sd)
    for i in range(1, D):
        for j in range(0, i):
            Ls[:, i, j] = np.random.normal(loc=0.0, scale=L_offdiag_sd, size=n).astype(np.float32)
    return Ls  # [n, D, D]

def _pack_vech_lower(L):
    # Pack lower-tri (row-major i>=j) into length D(D+1)/2
    n = L.shape[0]
    out = np.zeros((n, (D*(D+1))//2), dtype=np.float32)
    idx = 0
    for i in range(D):
        for j in range(i+1):
            out[:, idx] = L[:, i, j]
            idx += 1
    return out

def _simulate_case(mu, L, T):
    # X ~ N(mu, Sigma) with Sigma = L L^T
    Sigma = L @ L.T
    x = np.random.multivariate_normal(mean=mu.astype(np.float32),
                                      cov=Sigma.astype(np.float32),
                                      size=T).astype(np.float32)   # [T, D]
    if standardize_x:
        m = x.mean(axis=0, keepdims=True)
        s = x.std(axis=0, keepdims=True)
        s = np.where(s < 1e-9, 1.0, s)
        x = (x - m) / s
    return x

class MVNFullSimulator:
    def __init__(self, T_min, T_max):
        self.T_min, self.T_max = int(T_min), int(T_max)
        self._fixed_T = None

    def set_fixed_T(self, T=None):
        if T is None:
            self._fixed_T = None
        else:
            T = int(T)
            if not (self.T_min <= T <= self.T_max):
                raise ValueError(f'Fixed T={T} outside [{self.T_min}, {self.T_max}]')
            self._fixed_T = T

    def sample(self, n):
        # BayesFlow sometimes passes n as a tuple; be forgiving.
        n = int(n) if not isinstance(n, (tuple, list)) else int(n[0])
        T = self._fixed_T if self._fixed_T is not None else np.random.randint(self.T_min, self.T_max + 1)
        MU = _sample_mu(n)          # [n, D]
        Ls = _sample_L(n)           # [n, D, D]
        L_pack = _pack_vech_lower(Ls)  # [n, D(D+1)/2]

        # theta = concat([mu, vech(L)])
        theta = np.concatenate([MU, L_pack], axis=1).astype(np.float32)   # [n, K]

        X = np.empty((n, T, D), dtype=np.float32)
        for i in range(n):
            X[i] = _simulate_case(MU[i], Ls[i], T)

        out = {'theta': theta, 'trials': X, 'T': np.full((n,), T, dtype=np.int32)}
        if expose_split:
            out['mu']        = MU.astype('float32')
            out['L_packed']  = L_pack.astype('float32')
        return out

# Expose simulator and helpers
simulator = MVNFullSimulator(T_min, T_max)
def _bf_set_T(T):  simulator.set_fixed_T(T)
def _bf_clear_T(): simulator.set_fixed_T(None)

# Smoke test
_test = simulator.sample(3)
print('[sim ok] MVN-full keys:', list(_test.keys()),
      '| theta', _test['theta'].shape, '| trials', _test['trials'].shape, '| T', _test['T'][0])
", as.integer(min_trials), as.integer(max_trials),
                                    as.integer(D),
                                    mu_range[1], mu_range[2],
                                    L_diag_range[1], L_diag_range[2],
                                    L_offdiag_sd,
                                    if (standardize_x) "True" else "False",
                                    ex))
  invisible(TRUE)
}


# --- NEW: MVN means-only simulator (Σ = I) -------------------------------
bf_build_simulator_mvn_means_only_set <- function(
    min_trials        = 100L,
    max_trials        = 1000L,
    D                 = 3L,                 # data dimensionality
    mu_range          = c(-1.0, 1.0),       # per-dim uniform range for means
    standardize_x     = FALSE,               # usually FALSE here
    expose_split_keys = FALSE                # if TRUE, also expose 'mu'
) {
  stopifnot(D >= 1L, length(mu_range) == 2L)
  ex <- if (expose_split_keys) "True" else "False"
  
  reticulate::py_run_string(sprintf("
import numpy as np

# ------------ Config from R ------------
T_min, T_max = int(%d), int(%d)
D            = int(%d)
mu_min, mu_max = %f, %f
standardize_x  = %s
expose_split   = %s

def _sample_mu(n):
    # μ ~ U(mu_min, mu_max)^D
    MU = np.random.uniform(mu_min, mu_max, size=(n, D)).astype(np.float32)
    return MU

def _simulate_case(mu, T):
    # x ~ N(mu, I)
    z = np.random.normal(size=(T, D)).astype(np.float32)  # Σ = I
    x = z + mu[None, :]
    if standardize_x:
        m = x.mean(axis=0, keepdims=True)
        s = x.std(axis=0, keepdims=True)
        s = np.where(s < 1e-9, 1.0, s)
        x = (x - m) / s
    return x

class MVNMeansOnlySimulator:
    def __init__(self, T_min, T_max):
        self.T_min, self.T_max = int(T_min), int(T_max)
        self._fixed_T = None

    def set_fixed_T(self, T=None):
        if T is None:
            self._fixed_T = None
        else:
            T = int(T)
            if not (self.T_min <= T <= self.T_max):
                raise ValueError(f'Fixed T={T} outside [{self.T_min}, {self.T_max}]')
            self._fixed_T = T

    def sample(self, n):
        n = int(n) if not isinstance(n, (tuple, list)) else int(n[0])
        T = self._fixed_T if self._fixed_T is not None else np.random.randint(self.T_min, self.T_max + 1)
        MU = _sample_mu(n)                         # [n, D]
        X  = np.empty((n, T, D), dtype=np.float32) # [n, T, D]
        for i in range(n):
            X[i] = _simulate_case(MU[i], T)
        out = {'theta': MU, 'trials': X, 'T': np.full((n,), T, dtype=np.int32)}
        if expose_split:
            out['mu'] = MU.astype('float32')
        return out

# ------------ Expose `simulator` ------------
simulator = MVNMeansOnlySimulator(T_min, T_max)

# Convenience helpers for R (keep your API)
def _bf_set_T(T): simulator.set_fixed_T(T)
def _bf_clear_T(): simulator.set_fixed_T(None)

# Smoke test
_test = simulator.sample(3)
print('[sim ok] Means-only keys:', list(_test.keys()), '| theta', _test['theta'].shape, '| trials', _test['trials'].shape, '| T', _test['T'][0])
", as.integer(min_trials), as.integer(max_trials),
                                    as.integer(D),
                                    mu_range[1], mu_range[2],
                                    if (standardize_x) "True" else "False",
                                    ex))
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

# --- Helpers to name θ and reconstruct Σ ----------------------------------

# Names: mu1..muD only
mvn_mu_param_names <- function(D) paste0("mu", seq_len(D))

bf_add_mu_names <- function(df, D) {
  pcols <- grep("^p\\d+$", names(df), value = TRUE)
  stopifnot(length(pcols) == D)
  mu_names <- mvn_mu_param_names(D)
  
  # rename posterior
  names(df)[match(pcols, names(df))] <- mu_names
  
  # rename ground truth if present
  tcols <- grep("^true_p\\d+$", names(df), value = TRUE)
  if (length(tcols) == length(pcols)) {
    names(df)[match(tcols, names(df))] <- paste0("true_", mu_names)
  }
  df
}


mvn_param_names <- function(D) {
  # order: mu1..muD, then vech(L) with row-major lower-tri (i>=j)
  L_names <- unlist(lapply(seq_len(D), function(i)
    paste0("L", i, "_", seq_len(i))))
  c(paste0("mu", seq_len(D)), L_names)
}

# Extract L (DxD) from a numeric vector of length D + D(D+1)/2
mvn_unpack_L <- function(theta_row, D) {
  K <- D + (D*(D+1)) %/% 2
  stopifnot(length(theta_row) >= K)
  L <- matrix(0, nrow = D, ncol = D)
  idx <- D + 1L
  for (i in seq_len(D)) {
    for (j in seq_len(i)) {
      L[i, j] <- theta_row[idx]
      idx <- idx + 1L
    }
  }
  L
}

# Rename posterior columns p0.. to readable names; same for true_p*
bf_add_param_names <- function(df, D) {
  pcols   <- grep("^p\\d+$", names(df), value = TRUE)
  tcols   <- grep("^true_p\\d+$", names(df), value = TRUE)
  new_p   <- mvn_param_names(D)
  stopifnot(length(pcols) == length(new_p))
  names(df)[match(pcols, names(df))] <- new_p
  if (length(tcols) == length(pcols)) {
    names(df)[match(tcols, names(df))] <- paste0("true_", new_p)
  }
  df
}

# Given a posterior table (draw, case, mu*, L*), build Σ per (draw, case)
# Returns a data.frame with columns: draw, case, param, mean, q025, q975, true (if available), covered_95, bias
summarise_mvn_sigma_by_T <- function(df_post_named, D) {
  stopifnot(all(c("draw","case") %in% names(df_post_named)))
  # collect names
  mu_names <- paste0("mu", seq_len(D))
  L_names  <- setdiff(grep("^(mu|L)\\d", names(df_post_named), value = TRUE), mu_names)
  
  # build Σ per row
  uniq <- unique(df_post_named[, c("draw","case"), drop = FALSE])
  build_one <- function(dr, cs) {
    row <- df_post_named[df_post_named$draw == dr & df_post_named$case == cs, , drop = FALSE][1, ]
    th  <- as.numeric(unlist(row[c(mu_names, L_names)], use.names = FALSE))
    L   <- mvn_unpack_L(th, D)
    Sig <- tcrossprod(L, L)  # L %*% t(L)
    # true?
    has_true <- all(paste0("true_", c(mu_names, L_names)) %in% names(row))
    Sig_true <- NULL
    if (has_true) {
      th_true <- as.numeric(unlist(row[paste0("true_", c(mu_names, L_names))], use.names = FALSE))
      Ltrue   <- mvn_unpack_L(th_true, D)
      Sig_true <- tcrossprod(Ltrue, Ltrue)
    }
    list(Sig = Sig, Sig_true = Sig_true)
  }
  
  # Draw-wise Σ per case -> then summarise across draws, per case
  # Efficient route: loop by case, vectorize over draws
  cases <- sort(unique(df_post_named$case))
  out_list <- vector("list", length(cases))
  for (ci in seq_along(cases)) {
    cs <- cases[ci]
    rows <- df_post_named[df_post_named$case == cs, ]
    draws <- sort(unique(rows$draw))
    Sigs <- array(NA_real_, dim = c(length(draws), D, D))
    Sigs_true <- NULL
    for (k in seq_along(draws)) {
      dr <- draws[k]
      one <- build_one(dr, cs)
      Sigs[k,,] <- one$Sig
      if (!is.null(one$Sig_true)) Sigs_true <- one$Sig_true
    }
    # summarise variances and correlations
    var_names  <- paste0("sigma", seq_len(D), "_", seq_len(D))
    # pack lower-tri for correlations
    idx_lt <- which(lower.tri(matrix(NA, D, D)))
    # Variances
    var_df <- data.frame(
      param = paste0("Sigma[", seq_len(D), ",", seq_len(D), "]"),
      mean  = diag(apply(Sigs, c(2,3), mean))[seq_len(D)],
      q025  = diag(apply(Sigs, c(2,3), function(M) quantile(M, 0.025)))[seq_len(D)],
      q975  = diag(apply(Sigs, c(2,3), function(M) quantile(M, 0.975)))[seq_len(D)]
    )
    if (!is.null(Sigs_true)) {
      var_df$true <- diag(Sigs_true)
    }
    
    # Correlations
    Corrs <- array(NA_real_, dim = c(dim(Sigs)[1], D, D))
    for (k in seq_len(dim(Sigs)[1])) {
      Sd    <- sqrt(pmax(diag(Sigs[k,,]), 1e-12))
      Corrs[k,,] <- Sigs[k,,] / outer(Sd, Sd)
    }
    corr_pairs <- which(lower.tri(matrix(0, D, D)), arr.ind = TRUE)
    corr_df <- do.call(rbind, lapply(seq_len(nrow(corr_pairs)), function(r) {
      i <- corr_pairs[r,1]; j <- corr_pairs[r,2]
      v <- Corrs[, i, j]
      data.frame(
        param = sprintf("Corr[%d,%d]", i, j),
        mean  = mean(v),
        q025  = quantile(v, 0.025),
        q975  = quantile(v, 0.975),
        true  = if (!is.null(Sigs_true)) Sigs_true[i,j] / sqrt(Sigs_true[i,i]*Sigs_true[j,j]) else NA_real_
      )
    }))
    
    out_list[[ci]] <- transform(rbind(var_df, corr_df), case = cs)
  }
  out <- do.call(rbind, out_list)
  
  # coverage/bias if true available
  if ("true" %in% names(out)) {
    out$covered_95 <- with(out, ifelse(is.na(true), NA, (true >= q025) & (true <= q975)))
    out$bias       <- with(out, mean - true)
  }
  out
}


# Summaries for μ components only
bf_summarise_mu_by_T <- function(df_post_named, D) {
  stopifnot(all(c("T","draw","case") %in% names(df_post_named)))
  mu_cols <- mvn_mu_param_names(D)
  
  # 1) Long per-draw table
  long <- tidyr::pivot_longer(
    df_post_named,
    cols = tidyselect::all_of(mu_cols),
    names_to = "param",
    values_to = "value"
  )
  
  # 2) Summarise across draws first
  sumr <- long |>
    dplyr::group_by(T, case, param) |>
    dplyr::summarise(
      mean   = mean(value, na.rm = TRUE),
      median = median(value, na.rm = TRUE),
      q025   = quantile(value, 0.025, na.rm = TRUE),
      q975   = quantile(value, 0.975, na.rm = TRUE),
      .groups = "drop_last"
    ) |>
    dplyr::ungroup()
  
  # 3) Attach ground-truth once per (T, case, param)
  true_cols <- paste0("true_", mu_cols)
  if (all(true_cols %in% names(df_post_named))) {
    true_tbl <- df_post_named |>
      dplyr::select(T, case, tidyselect::all_of(true_cols)) |>
      dplyr::distinct() |>
      tidyr::pivot_longer(
        cols = tidyselect::all_of(true_cols),
        names_to = "param_true", values_to = "true"
      ) |>
      dplyr::mutate(param = sub("^true_", "", param_true)) |>
      dplyr::select(-param_true)
    
    sumr <- dplyr::left_join(sumr, true_tbl, by = c("T","case","param")) |>
      dplyr::mutate(
        covered_95 = ifelse(is.na(true), NA, (true >= q025) & (true <= q975)),
        bias       = mean - true
      )
  }
  
  sumr
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

save_recovery_pages_by_T_mu <- function(sumr_T, D,
                                        y_stat   = "median",
                                        out_dir  = "plots",
                                        filename = "recovery_mu_by_T.pdf") {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  Ts <- sort(unique(sumr_T$T))
  pdf_path <- file.path(out_dir, filename)
  
  grDevices::pdf(pdf_path, width = 10, height = 8)
  on.exit(grDevices::dev.off(), add = TRUE)
  
  mu_levels <- mvn_mu_param_names(D)
  
  for (Ti in Ts) {
    dTi <- sumr_T[sumr_T$T == Ti, , drop = FALSE]
    out <- bf_plot_recovery(
      dTi,
      param_order = mu_levels,
      y_stat = y_stat
    )
    p <- out$patchwork + patchwork::plot_annotation(
      title = sprintf("μ recovery at T = %d (Σ = I)", Ti)
    )
    print(p)
  }
  message(sprintf("[saved] %s with %d pages.", pdf_path, length(Ts)))
  invisible(pdf_path)
}


# ---- Unpack helpers -------------------------------------------------------

# Names we expect after bf_add_param_names(D):
# mu1..muD, L1_1, L2_1, L2_2, L3_1, L3_2, L3_3, ...
mvn_param_names <- function(D) {
  L_names <- unlist(lapply(seq_len(D), function(i) paste0("L", i, "_", seq_len(i))))
  c(paste0("mu", seq_len(D)), L_names)
}

# Given a numeric named vector with mu* and L* (or a 1-row data.frame),
# return list(mu, L, Sigma)
theta_unpack_mvn <- function(x, D) {
  if (is.data.frame(x)) x <- unlist(x[1, , drop = TRUE], use.names = TRUE)
  stopifnot(is.numeric(x), !is.null(names(x)))
  mu_names <- paste0("mu", seq_len(D))
  L_names  <- setdiff(mvn_param_names(D), mu_names)
  
  stopifnot(all(mu_names %in% names(x)),
            all(L_names  %in% names(x)))
  
  mu <- as.numeric(x[mu_names])
  L  <- matrix(0, D, D)
  idx <- 1L
  for (i in seq_len(D)) {
    for (j in seq_len(i)) {
      L[i, j] <- as.numeric(x[L_names[idx]])
      idx <- idx + 1L
    }
  }
  Sigma <- L %*% t(L)
  list(mu = mu, L = L, Sigma = Sigma)
}

# Build posterior arrays of Σ across draws and cases
# Input: df_post_named has columns draw, case, mu*, L* (already renamed)
# Output: list(Sigma = [draw, case, D, D], Sigma_true = [case, D, D] or NULL)
posterior_sigma_array <- function(df_post_named, D) {
  stopifnot(all(c("draw","case") %in% names(df_post_named)))
  mu_names <- paste0("mu", seq_len(D))
  L_names  <- setdiff(mvn_param_names(D), mu_names)
  
  draws <- sort(unique(df_post_named$draw))
  cases <- sort(unique(df_post_named$case))
  
  Sigs <- array(NA_real_, dim = c(length(draws), length(cases), D, D),
                dimnames = list(NULL, NULL, NULL, NULL))
  
  # optional truth
  t_mu_names <- paste0("true_", mu_names)
  t_L_names  <- paste0("true_", L_names)
  have_truth <- all(c(t_mu_names, t_L_names) %in% names(df_post_named))
  Sig_true <- if (have_truth) array(NA_real_, dim = c(length(cases), D, D)) else NULL
  
  for (ci in seq_along(cases)) {
    cs <- cases[ci]
    rows_cs <- df_post_named[df_post_named$case == cs, , drop = FALSE]
    rows_cs <- rows_cs[order(rows_cs$draw), , drop = FALSE]
    
    for (di in seq_along(draws)) {
      row <- rows_cs[rows_cs$draw == draws[di], , drop = FALSE]
      th  <- c(unlist(row[mu_names]), unlist(row[L_names]))
      Sigs[di, ci, , ] <- theta_unpack_mvn(th, D)$Sigma
    }
    
    if (have_truth) {
      th_true <- c(unlist(rows_cs[1, t_mu_names]), unlist(rows_cs[1, t_L_names]))
      Sig_true[ci, , ] <- theta_unpack_mvn(th_true, D)$Sigma
    }
  }
  
  list(Sigma = Sigs, Sigma_true = Sig_true)
}

bf_plot_sigma_recovery <- function(sumr_sigma,
                                   type = c("violin","heatmap"),
                                   D,
                                   T_value = NULL,
                                   case = NULL) {
  type <- match.arg(type)
  stopifnot(requireNamespace("ggplot2", quietly = TRUE),
            requireNamespace("dplyr", quietly = TRUE),
            requireNamespace("tidyr", quietly = TRUE),
            requireNamespace("patchwork", quietly = TRUE))
  library(ggplot2); library(dplyr); library(tidyr)
  
  if (type == "violin") {
    # Split variances vs correlations
    is_var  <- grepl("^Sigma\\[\\d+,\\1\\]$|^Sigma\\[(\\d+),(\\1)\\]$", sumr_sigma$param, perl = TRUE) |
      grepl("^Sigma\\[(\\d+),(\\1)\\]$", sumr_sigma$param, perl = TRUE) |
      grepl("^Sigma\\[\\d+,\\d+\\]$", sumr_sigma$param) & grepl(",\\1\\]", sumr_sigma$param)
    # simpler: treat diagonal by regex
    is_var  <- grepl("^Sigma\\[(\\d+),(\\1)\\]$", sumr_sigma$param, perl = TRUE)
    df_var  <- sumr_sigma[is_var, , drop = FALSE]
    df_corr <- sumr_sigma[grepl("^Corr\\[\\d+,\\d+\\]$", sumr_sigma$param), , drop = FALSE]
    
    # Variances violin (by param) faceted by T
    p_var <- ggplot(df_var, aes(x = param, y = mean)) +
      geom_violin(alpha = 0.6, trim = FALSE) +
      geom_boxplot(width = 0.15, outlier.shape = NA) +
      geom_point(aes(y = true), color = "red", shape = 4, stroke = 1.1, na.rm = TRUE) +
      facet_wrap(~ T, scales = "free_y") +
      labs(title = "Variance recovery", x = NULL, y = "Posterior mean of Sigma[i,i]") +
      theme_classic() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    # Correlations violin (by pair) faceted by T
    p_corr <- ggplot(df_corr, aes(x = param, y = mean)) +
      geom_violin(alpha = 0.6, trim = FALSE) +
      geom_boxplot(width = 0.15, outlier.shape = NA) +
      geom_point(aes(y = true), color = "red", shape = 4, stroke = 1.1, na.rm = TRUE) +
      facet_wrap(~ T) +
      labs(title = "Correlation recovery", x = NULL, y = "Posterior mean of Corr[i,j]") +
      theme_classic() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    # Coverage bar by parameter (if present)
    have_cov <- "covered_95" %in% names(sumr_sigma)
    if (have_cov) {
      cov_tbl <- sumr_sigma %>%
        group_by(T, param) %>%
        summarise(coverage_95 = mean(covered_95, na.rm = TRUE), .groups = "drop")
      p_cov <- ggplot(cov_tbl, aes(x = param, y = coverage_95)) +
        geom_col(fill = "skyblue", color = "white") +
        geom_hline(yintercept = 0.95, linetype = 2) +
        facet_wrap(~ T) +
        coord_cartesian(ylim = c(0,1)) +
        labs(title = "95% CI coverage", x = NULL, y = "Coverage") +
        theme_classic() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
      return(list(plots = list(variances = p_var, correlations = p_corr, coverage = p_cov),
                  patchwork = (p_var / p_corr / p_cov) + patchwork::plot_layout(heights = c(1,1,0.8))))
    } else {
      return(list(plots = list(variances = p_var, correlations = p_corr),
                  patchwork = (p_var / p_corr)))
    }
  }
  
  # type == "heatmap"
  stopifnot(!is.null(T_value), !is.null(case))
  df_Tc <- sumr_sigma %>% filter(T == T_value) %>% filter(case == case)
  
  # Need the full matrices; rebuild from parameter rows:
  # Variances (diagonal)
  diag_idx <- which(grepl("^Sigma\\[(\\d+),(\\1)\\]$", df_Tc$param, perl = TRUE))
  # Correlations (lower-tri)
  corr_rows <- df_Tc %>% filter(grepl("^Corr\\[\\d+,\\d+\\]$", param))
  
  # Build mean Sigma via diag & corr
  Sig_mean <- matrix(NA_real_, D, D)
  # fill diagonal means
  for (i in seq_len(D)) {
    rowi <- df_Tc %>% filter(param == sprintf("Sigma[%d,%d]", i, i))
    Sig_mean[i, i] <- rowi$mean[1]
  }
  # fill off-diagonals from correlations and variances
  for (i in seq_len(D)) for (j in seq_len(D)) if (i > j) {
    cij <- corr_rows %>% filter(param == sprintf("Corr[%d,%d]", i, j))
    if (nrow(cij) == 1) {
      sdi <- sqrt(Sig_mean[i,i]); sdj <- sqrt(Sig_mean[j,j])
      Sig_mean[i, j] <- cij$mean[1] * sdi * sdj
      Sig_mean[j, i] <- Sig_mean[i, j]
    }
  }
  
  # True Sigma if present
  has_true <- "true" %in% names(df_Tc)
  Sig_true <- if (has_true) {
    S <- matrix(NA_real_, D, D)
    for (i in seq_len(D)) S[i,i] <- (df_Tc %>% filter(param == sprintf("Sigma[%d,%d]", i, i)))$true[1]
    for (i in seq_len(D)) for (j in seq_len(D)) if (i > j) {
      cij <- (df_Tc %>% filter(param == sprintf("Corr[%d,%d]", i, j)))$true[1]
      sdi <- sqrt(S[i,i]); sdj <- sqrt(S[j,j])
      S[i, j] <- cij * sdi * sdj
      S[j, i] <- S[i, j]
    }
    S
  } else NULL
  
  # Long for heatmap
  to_long <- function(M, lab) {
    as.data.frame(as.table(M)) |>
      dplyr::rename(i = Var1, j = Var2, val = Freq) |>
      dplyr::mutate(which = lab)
  }
  df_hm <- to_long(Sig_mean, "Posterior mean")
  if (!is.null(Sig_true)) df_hm <- dplyr::bind_rows(df_hm, to_long(Sig_true, "True"))
  
  p_hm <- ggplot(df_hm, aes(x = j, y = i, fill = val)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.2f", val)), size = 3) +
    scale_y_reverse() + coord_fixed() +
    facet_wrap(~ which) +
    labs(title = sprintf("Sigma heatmaps (T=%d, case=%d)", T_value, case),
         x = "j", y = "i", fill = "Σ") +
    theme_minimal()
  
  list(plots = list(heatmap = p_hm), patchwork = p_hm)
}


# -------------------------------------------------------------------------


# -------------------------------------------------------------------------

D <- 4  # example
bf_build_simulator_mvn_full_set(
  min_trials        = 100,
  max_trials        = 1000,
  D                 = D,
  mu_range          = c(-1, 1),
  L_diag_range      = c(0.4, 2.0),
  L_offdiag_sd      = 0.25,
  standardize_x     = FALSE,
  expose_split_keys = TRUE
)

bf_build_workflow_deepset(
  phi_units = c(128, 128),
  rho_units = c(128, 128),
  aggregation = "mean",
  flow_type = "coupling",
  n_flow_layers = 6,
  flow_hidden = c(128, 128),
  param_keys = "theta",   # theta now = [mu, vech(L)]
  data_key  = "trials"
)

bf_fit_online(
  epochs = 3, num_batches_per_epoch = 50, batch_size = 256,
  with_validation = TRUE, N_val = 400
)

# Draw posteriors for a couple of T values
df_post_T <- bf_sample_posteriors_by_T(
  T_values     = c(500L),
  cases_per_T  = 150L,
  num_samples  = 2000L
)

# Rename p* to (mu1..muD, L*) and true_p* likewise
df_post_T <- bf_add_param_names(df_post_T, D)

# --- Summaries for mu ---
sumr_mu <- bf_summarise_mu_by_T(df_post_T, D)

# --- Summaries for Sigma (built from L) ---
# sumr_sigma <- summarise_mvn_sigma_by_T(df_post_T, D)

# Example: save recovery pages for mu (works with your existing saver)
save_recovery_pages_by_T_mu(sumr_mu, D, y_stat = "median",
                            out_dir = "plots",
                            filename = "recovery_mu_by_T.pdf")

# Assuming you already have:
# df_post_T <- bf_sample_posteriors_by_T(...);
# df_post_T <- bf_add_param_names(df_post_T, D)
sumr_sigma <- summarise_mvn_sigma_by_T(df_post_T, D)

# 1) Violin dashboards across T
out_violin <- bf_plot_sigma_recovery(sumr_sigma, type = "violin", D = D)
out_violin$patchwork  # display

# 2) Heatmaps for a specific (T, case)
T_pick   <- sort(unique(sumr_sigma$T))[1]
case_pick <- 0  # or any case id present
out_hm <- bf_plot_sigma_recovery(sumr_sigma, type = "heatmap", D = D,
                                 T_value = T_pick, case = case_pick)
out_hm$patchwork


# -------------------------------------------------------------------------

# # 1) Build μ-only simulator (Σ = I)
# D <- 4
# bf_build_simulator_mvn_means_only_set(
#   min_trials    = 400,
#   max_trials    = 4000,
#   D             = D,
#   mu_range      = c(-1, 1),
#   standardize_x = FALSE,
#   expose_split_keys = TRUE      # optional; exposes 'mu' too
# )
# 
# # 2) Build workflow (unchanged)
# bf_build_workflow_deepset(
#   phi_units = c(128, 128),
#   rho_units = c(128, 128),
#   aggregation = "mean",
#   flow_type = "coupling",
#   n_flow_layers = 6,
#   flow_hidden = c(128, 128),
#   param_keys = "theta",          # now θ has shape [N, D] with μ only
#   data_key  = "trials"
# )
# 
# # 3) Train (unchanged)
# bf_fit_online(
#   epochs = 3, num_batches_per_epoch = 50, batch_size = 256,
#   with_validation = TRUE, N_val = 400
# )
# 
# # 4) Draw posteriors for multiple T, then name columns as μ1..μD
# df_post_T <- bf_sample_posteriors_by_T(
#   T_values     = c(400L, 4000L),
#   cases_per_T  = 100L,
#   num_samples  = 2000L
# )
# df_post_T <- bf_add_mu_names(df_post_T, D)
# 
# # 5) Summarise by T for μ only
# sumr_T    <- bf_summarise_mu_by_T(df_post_T, D)
# 
# # 6) Make a single multi-page PDF (one page per T)
# save_recovery_pages_by_T_mu(sumr_T, D, y_stat = "median",
#                             out_dir = "plots",
#                             filename = "recovery_mu_by_T.pdf")
