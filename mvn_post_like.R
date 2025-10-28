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


# --- Global mode holder ---------------------------------------------------
BF_MODE <- "posterior"   # or "likelihood"

bf_set_mode <- function(mode = c("posterior","likelihood")) {
  mode <- match.arg(mode)
  assign("BF_MODE", mode, inherits = TRUE)
  invisible(mode)
}

# --- Python helpers for MVN stats (mean + sample covariance -> vech) -----
reticulate::py_run_string("
import numpy as np

def _vech_lower(M):
    D = M.shape[-1]
    out = []
    for i in range(D):
        for j in range(i+1):
            out.append(M[..., i, j])
    return np.stack(out, axis=-1)

def mvn_stats_from_trials(X, eps=1e-6):
    # X: [N, T, D]
    N, T, D = X.shape
    mean = X.mean(axis=1)                                 # [N,D]
    # sample covariance per case
    Xc = X - mean[:,None,:]
    # unbiased denom (T-1); guard for T==1
    denom = np.maximum(T-1, 1)
    cov = (Xc.transpose(0,2,1) @ Xc) / denom              # [N,D,D]
    # jitter for numeric stability
    cov = cov + eps*np.eye(D, dtype=X.dtype)[None,:,:]
    vech = _vech_lower(cov)                               # [N, D(D+1)/2]
    stats = np.concatenate([mean, vech], axis=-1).astype(np.float32)
    return stats
")


# MVN (mu, Sigma) simulator using np.random.multivariate_normal ----------
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


# Switchable workflow builder -----------------------------------------
bf_build_workflow_mvn_switchable <- function(
    mode         = c("posterior","likelihood"),
    # DeepSet (posterior)
    phi_units    = c(256,256,128),
    rho_units    = c(256,128),
    aggregation  = c("mean","sum","max"),
    # Flow
    flow_type    = c("coupling","autoregressive"),
    n_flow_layers= 8L,
    flow_hidden  = c(256,256),
    # Data keys from your simulators
    param_keys   = "theta",     # theta = [mu, vech(L)] or just mu
    data_key     = "trials"     # set of trials X with shape [T,D]
) {
  aggregation <- match.arg(aggregation)
  flow_type   <- match.arg(flow_type)
  mode        <- match.arg(mode)
  if (exists("bf_set_mode", mode = "function")) bf_set_mode(mode)
  
  # Ensure a Python simulator exists (_sim_obj)
  reticulate::py_run_string("
import builtins
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
  
  # Build the flow net code
  flowhid <- paste0("[", paste(as.integer(flow_hidden), collapse=","), "]")
  flow_code <-
    if (flow_type == "coupling") {
      sprintf("inference_net = bf.networks.CouplingFlow(n_coupling_layers=%d, hidden_units=%s)",
              as.integer(n_flow_layers), flowhid)
    } else {
      sprintf("inference_net = bf.networks.AutoregressiveFlow(n_layers=%d, hidden_units=%s)",
              as.integer(n_flow_layers), flowhid)
    }
  
  if (mode == "posterior") {
    # Posterior: learn p(theta | X)
    phi_py <- paste0("[", paste(as.integer(phi_units), collapse=","), "]")
    rho_py <- paste0("[", paste(as.integer(rho_units), collapse=","), "]")
    reticulate::py_run_string(sprintf("
import numpy as np, bayesflow as bf

adapter = (bf.Adapter()
            .convert_dtype('float64','float32')
            .rename('%s','summary_variables')      # X -> summary_variables
            .rename('%s','inference_variables'))   # theta -> inference_variables

summary_net = bf.networks.DeepSet(phi_units=%s, rho_units=%s, aggregation='%s')
try:
    %s
except TypeError:
    inference_net = bf.networks.CouplingFlow()

workflow = bf.BasicWorkflow(
    simulator=_sim_obj,
    adapter=adapter,
    summary_network=summary_net,
    inference_network=inference_net,
    inference_variables='inference_variables',
    summary_variables='summary_variables'
)

_raw = _sim_obj.sample(4)
_bat = adapter(_raw)
import numpy as np
shapes = {k: (np.array(v).shape if hasattr(v,'__array__') else getattr(v,'shape',None)) for k,v in _bat.items()}
print('[ok] switchable workflow ready; mode= posterior ; adapted shapes:', shapes)
", data_key, param_keys, phi_py, rho_py, aggregation, flow_code))

  } else {
    # Likelihood: learn p(S(X) | theta)
    reticulate::py_run_string(sprintf("
import numpy as np, bayesflow as bf

def _lik_adapter_canonical(raw, *args, **kwargs):
    X  = raw['%s']                                # [N,T,D]
    th = raw['%s']                                # [N,K_theta]
    stats = mvn_stats_from_trials(X)              # [N,K_stats]
    return {
        'inference_variables':  stats.astype('float32'),
        'inference_conditions': th.astype('float32')
    }

adapter = _lik_adapter_canonical
try:
    %s
except TypeError:
    inference_net = bf.networks.CouplingFlow()

workflow = bf.BasicWorkflow(
    simulator=_sim_obj,
    adapter=adapter,
    inference_network=inference_net,
    inference_variables='inference_variables',     # stats
    inference_conditions='inference_conditions',   # theta
    standardize=None
)

_raw = _sim_obj.sample(4)
_bat = adapter(_raw)
import numpy as np
shapes = {k: (np.array(v).shape if hasattr(v,'__array__') else getattr(v,'shape',None)) for k,v in _bat.items()}
print('[ok] switchable workflow ready; mode= likelihood ; adapted shapes:', shapes)
", data_key, param_keys, flow_code))
  }
  
  invisible(TRUE)
}



# -------------------------------------------------------------------------

# Posterior sampler (your existing one) still fine for BF_MODE == "posterior".
# For likelihood mode, use the tools below instead.

# Evaluate learned log-likelihood of stats for a batch of cases vs a theta grid
# - Draw N_test cases from simulator
# - For each case c, compute S_c = stats(X_c)
# - Evaluate log p_phi(S_c | theta) on a theta grid
# Replace your likelihood helpers with these key-robust, canonical-key versions ----

bf_eval_likelihood_on_grid <- function(theta_grid, N_test = 64L) {
  stopifnot(BF_MODE == "likelihood")
  stopifnot(is.matrix(theta_grid))
  py$TH_GRID <- as.matrix(theta_grid)
  
  # Sanity: infer K_theta from simulator once and check grid width
  reticulate::py_run_string("
_rawK = _sim_obj.sample(1)
if 'theta' not in _rawK:
    raise KeyError(\"Simulator output must include 'theta' for likelihood mode.\")
K_theta = int(_rawK['theta'].shape[1])
")
  K_theta <- as.integer(reticulate::py$K_theta)
  if (ncol(theta_grid) != K_theta) {
    stop(sprintf('theta_grid has %d columns but simulator theta has K=%d. Fix grid width.',
                 ncol(theta_grid), K_theta))
  }
  
  # Evaluate log p_phi(S(X)|theta) on a grid, using canonical keys
  reticulate::py_run_string(sprintf("
import numpy as np
_raw = _sim_obj.sample(%d)

# robust data-key detection
if 'trials' in _raw:
    X = _raw['trials']
elif 'x' in _raw:
    X = _raw['x']
else:
    raise KeyError('Expected key \\'trials\\' or \\'x\\' in simulator output.')

S = mvn_stats_from_trials(np.array(X, dtype='float32'))   # [N, K_stats]
G = int(np.array(TH_GRID).shape[0]); N = int(S.shape[0])

stats = np.repeat(S, repeats=G, axis=0)                   # [N*G, K_stats]
theta = np.tile(np.array(TH_GRID, dtype='float32'), (N,1))# [N*G, K_theta]

LL = workflow.log_prob(data={
    'inference_variables':  stats,
    'inference_conditions': theta
})
LL = LL.reshape(N, G)
lik_eval = dict(loglik=LL, stats=S, theta_grid=np.array(TH_GRID))
"))
  list(
    loglik      = reticulate::py$lik_eval$loglik,
    stats       = reticulate::py$lik_eval$stats,
    theta_grid  = theta_grid
  )
}

bf_loglik_for_observed <- function(X_obs, theta_grid) {
  stopifnot(BF_MODE == "likelihood")
  stopifnot(is.matrix(X_obs))
  stopifnot(is.matrix(theta_grid))
  py$X_OBS   <- array(X_obs, dim = c(1L, nrow(X_obs), ncol(X_obs)))
  py$TH_GRID <- as.matrix(theta_grid)
  
  # check K consistency
  reticulate::py_run_string("
_rawK = _sim_obj.sample(1)
K_theta = int(_rawK['theta'].shape[1])
")
  K_theta <- as.integer(reticulate::py$K_theta)
  if (ncol(theta_grid) != K_theta) {
    stop(sprintf('theta_grid has %d columns but simulator theta has K=%d. Fix grid width.',
                 ncol(theta_grid), K_theta))
  }
  
  reticulate::py_run_string("
import numpy as np
S = mvn_stats_from_trials(np.array(X_OBS, dtype='float32'))   # [1, K_stats]
G = int(np.array(TH_GRID).shape[0])

stats = np.repeat(S, repeats=G, axis=0)
theta = np.array(TH_GRID, dtype='float32')

LL = workflow.log_prob(data={
    'inference_variables':  stats,
    'inference_conditions': theta
})
loglik_obs = LL.reshape(1, G)
")
  as.numeric(reticulate::py$loglik_obs[1, ])
}

bf_sample_stats_given_theta <- function(num_samples = 2000L, theta0) {
  stopifnot(BF_MODE == "likelihood")
  py$TH0 <- matrix(theta0, nrow = 1)
  reticulate::py_run_string(sprintf("
import numpy as np
samps = workflow.sample(
    num_samples=%d,
    conditions={'inference_conditions': np.array(TH0, dtype='float32')}
)
S = None
if isinstance(samps, dict):
    for k in ['inference_variables','stats','x','y','z']:
        if k in samps:
            S = samps[k][0]
            break
if S is None:
    S = samps
lik_samps = np.asarray(S)
", as.integer(num_samples)))
reticulate::py$lik_samps
}



# -------------------------------------------------------------------------

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

# Amortise over T by randomising the set size per call, and allow pinning T.
bf_set_T   <- function(T) reticulate::py_run_string(sprintf("_bf_set_T(%d)", as.integer(T)))
bf_clear_T <- function()  reticulate::py_run_string("_bf_clear_T()")

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

# Infer D from p* columns (K = D + D(D+1)/2) --------------------------
bf_infer_D_from_pcols <- function(df) {
  pcols <- grep("^p\\d+$", names(df), value = TRUE)
  K <- length(pcols)
  D <- (-3 + sqrt(9 + 8*K)) / 2
  if (!is.finite(D) || abs(D - round(D)) > 1e-8 || D < 1)
    stop(sprintf("Cannot infer integer D from K=%d p-columns.", K))
  as.integer(round(D))
}

# Ensure mu1..muD exist
bf_ensure_mu_names <- function(df, D = NULL) {
  has_mu <- any(grepl("^mu\\d+$", names(df)))
  if (!has_mu) {
    # if D missing/invalid, infer from p* columns
    if (is.null(D) || !is.numeric(D) || length(D)!=1L || D < 1) {
      D <- bf_infer_D_from_pcols(df)
    } else {
      D <- as.integer(D)
    }
    df <- bf_add_param_names_safe(df, D)
  }
  list(df = df, D = sum(grepl("^mu\\d+$", names(df))))
}

# Save one page per T (auto-infers D) 
save_recovery_pages_by_T_mu_safe <- function(sumr_T,
                                             D = NULL,
                                             y_stat   = "median",
                                             out_dir  = "plots",
                                             filename = "recovery_mu_by_T.pdf",
                                             pad_x = 0.005,
                                             pad_y = 0.005) {
  # Require pkgs
  if (!requireNamespace("ggplot2", quietly = TRUE) ||
      !requireNamespace("patchwork", quietly = TRUE))
    stop("Please install ggplot2 and patchwork.")
  
  # If sumr_T isn’t from bf_summarise_mu_by_T_safe, infer D from columns
  if (is.null(D) || !is.numeric(D) || length(D)!=1L || D < 1) {
    D <- sum(grepl("^mu\\d+$", names(sumr_T)))  # summaries usually keep 'param' not mu*, so:
    if (D == 0L && "param" %in% names(sumr_T)) {
      # infer from distinct param names like "mu1","mu2",...
      D <- sum(grepl("^mu\\d+$", unique(sumr_T$param)))
    }
    if (D == 0L) stop("Could not infer D from `sumr_T`; ensure it includes mu-param rows.")
  }
  D <- as.integer(D)
  
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  pdf_path <- file.path(out_dir, filename)
  
  # ensure param levels/order
  mu_levels <- paste0("mu", seq_len(D))
  if ("param" %in% names(sumr_T)) {
    sumr_T$param <- factor(sumr_T$param, levels = mu_levels)
  }
  
  # plotting (uses your existing bf_plot_recovery)
  grDevices::pdf(pdf_path, width = 10, height = 8)
  on.exit(grDevices::dev.off(), add = TRUE)
  
  Ts <- sort(unique(sumr_T$T))
  for (Ti in Ts) {
    dTi <- sumr_T[sumr_T$T == Ti, , drop = FALSE]
    out <- bf_plot_recovery(
      dTi,
      param_order = mu_levels,
      y_stat = y_stat,
      pad_x = pad_x,
      pad_y = pad_y
    )
    p <- out$patchwork + patchwork::plot_annotation(
      title = sprintf("μ recovery at T = %d", Ti)
    )
    print(p)
  }
  message(sprintf("[saved] %s with %d pages.", pdf_path, length(Ts)))
  invisible(pdf_path)
}

mvn_param_names <- function(D) {
  # order: mu1..muD, then vech(L) with row-major lower-tri (i>=j)
  L_names <- unlist(lapply(seq_len(D), function(i)
    paste0("L", i, "_", seq_len(i))))
  c(paste0("mu", seq_len(D)), L_names)
}

# Infer D from number of p* columns (K = D + D(D+1)/2)
bf_infer_D <- function(df) {
  pcols <- grep("^p\\d+$", names(df), value = TRUE)
  K <- length(pcols)
  D <- (-3 + sqrt(9 + 8*K)) / 2
  if (!is.finite(D) || abs(D - round(D)) > 1e-8 || D < 1)
    stop(sprintf("Cannot infer integer D from K=%d p-columns.", K))
  as.integer(round(D))
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

# Safe renamer (uses D if valid, else infers)
bf_add_param_names_safe <- function(df, D = NULL) {
  if (is.null(D) || !is.numeric(D) || length(D) != 1L || D < 1) {
    D <- bf_infer_D(df)
  } else {
    D <- as.integer(D)
    K_expected <- D + D*(D+1)/2
    pcols <- grep("^p\\d+$", names(df), value = TRUE)
    if (length(pcols) != K_expected)
      stop(sprintf("For D=%d expected %d p-columns, found %d.",
                   D, K_expected, length(pcols)))
  }
  bf_add_param_names(df, D)
}

# Safe μ-summary (will rename if needed + infer D)
bf_summarise_mu_by_T_safe <- function(df_post) {
  # Ensure mu* columns exist; if not, infer D and rename
  if (!any(grepl("^mu\\d+$", names(df_post)))) {
    df_post <- bf_add_param_names_safe(df_post)
  }
  # Now D = number of mu* cols
  D <- sum(grepl("^mu\\d+$", names(df_post)))
  bf_summarise_mu_by_T(df_post, D)
}

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


# Posterior learning ------------------------------------------------------

D <- 3  # example
bf_build_simulator_mvn_full_set(
  min_trials        = 100,
  max_trials        = 1000,
  D                 = D,
  mu_range          = c(-1, 1),
  L_diag_range      = c(0.5, 1.0),
  L_offdiag_sd      = 0.1,
  standardize_x     = FALSE,
  expose_split_keys = TRUE
)

# Build workflow in posterior mode
bf_build_workflow_mvn_switchable(
  mode = "posterior",
  phi_units = c(128,128),
  rho_units = c(128,128),
  aggregation = "mean",
  flow_type = "coupling",
  n_flow_layers = 6,
  flow_hidden = c(128,128),
  param_keys = "theta",
  data_key  = "trials"
)

# Run training
bf_fit_online(epochs = 2, num_batches_per_epoch = 40, batch_size = 256,
              with_validation = TRUE, N_val = 400)

# Save

# Load

# Your existing posterior pipeline keeps working:
df_post_T <- bf_sample_posteriors_by_T(T_values = c(400L),
                                       cases_per_T = 120L,
                                       num_samples = 2000L)
df_post_T <- bf_add_param_names_safe(df_post_T)
sumr_mu   <- bf_summarise_mu_by_T_safe(df_post_T)
save_recovery_pages_by_T_mu_safe(
  sumr_T  = sumr_mu,      # your summary tibble
  D       = NULL,         # let it infer
  y_stat  = "median",
  out_dir = "plots",
  filename = "recovery_mu_by_T.pdf"
)


# CHECK THIS --------------------------------------------------------------
# sumr_sigma <- summarise_mvn_sigma_by_T(df_post_T, D)


# Likelihood learning -----------------------------------------------------

# Switch to likelihood mode (learn density over stats given theta)
bf_build_workflow_mvn_switchable(
  mode = "likelihood",
  flow_type = "coupling",
  n_flow_layers = 6,
  flow_hidden = c(128,128),
  param_keys = "theta",
  data_key  = "trials"
)

bf_fit_online(epochs = 2, num_batches_per_epoch = 40, batch_size = 256,
              with_validation = TRUE, N_val = 400)

# Evaluate learned log-likelihood curves over a theta grid for random test cases
# theta_grid must match your theta layout: [mu(1..D), vech(L)]
# Example: reuse true thetas from the simulator to build a grid around them,
# or just sample a Latin hypercube; here we show a generic call:
D = 3
theta_grid <- matrix(rnorm(200 * (D + D*(D+1)/2)), nrow = 200)  # placeholder
lik_res <- bf_eval_likelihood_on_grid(theta_grid, N_test = 32)

# For a specific observed dataset X_obs [T,D], get log-lik curve on the same grid:
# X_obs <- <your T x D matrix>
ll_curve <- bf_loglik_for_observed(X_obs, theta_grid)

# Draw samples of stats given a fixed theta0:
theta0 <- theta_grid[1, ]
S_samps <- bf_sample_stats_given_theta(2000, theta0)

