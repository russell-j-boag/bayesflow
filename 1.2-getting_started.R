rm(list = ls())

# https://kucharssim.github.io/bayesflow-cognitive-modeling-book/bayesian-cognitive-modeling/01-getting-started/02-getting-started-with-bayesflow.html

# Run in terminal:
# source /Users/rjb779/Library/r-miniconda-arm64/etc/profile.d/conda.sh
# conda activate r-bf
# 
# /Users/rjb779/Library/r-miniconda-arm64/envs/r-bf/bin/python -m pip uninstall -y tensorflow
# /Users/rjb779/Library/r-miniconda-arm64/envs/r-bf/bin/python -m pip uninstall -y tensorflow-macos
# /Users/rjb779/Library/r-miniconda-arm64/envs/r-bf/bin/python -m pip uninstall -y tensorflow-metal
# 
# /Users/rjb779/Library/r-miniconda-arm64/envs/r-bf/bin/python -m pip install "numpy<2.0" "scipy>=1.13,<1.16"
# /Users/rjb779/Library/r-miniconda-arm64/envs/r-bf/bin/python -m pip install -U 'jax[cpu]' keras
# /Users/rjb779/Library/r-miniconda-arm64/envs/r-bf/bin/python -m pip install -U --no-deps bayesflow pandas matplotlib seaborn
# 
# conda env config vars set KERAS_BACKEND=jax
# conda deactivate
# conda activate r-bf
# 
# /Users/rjb779/Library/r-miniconda-arm64/envs/r-bf/bin/python - <<'PY'
# import os, keras, numpy, scipy, jax
# print("KERAS_BACKEND env:", os.environ.get("KERAS_BACKEND"))
# print("Keras backend    :", keras.config.backend())
# print("NumPy            :", numpy.__version__)
# print("SciPy            :", scipy.__version__)
# print("JAX              :", jax.__version__)
# PY

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

# Quick test that pyplot works
py_run_string("
import numpy as np, matplotlib.pyplot as plt
x = np.linspace(0, 2*np.pi, 200)
plt.figure()
plt.plot(x, np.sin(x), label='sin')
plt.legend(); plt.tight_layout()
# If using Agg (headless), this will save a file; otherwise a window may pop up
plt.savefig('pyplot_smoke.png', dpi=150)
print('Saved: pyplot_smoke.png')
")


# Create simulator --------------------------------------------------------

py_run_string("
def prior():
    return {'theta': np.random.beta(a=1.0, b=1.0)}

def likelihood(theta):
    # ensure plain float for binomial 'p'
    return {'k': np.random.binomial(n=10, p=float(theta))}

simulator = bf.simulators.make_simulator([prior, likelihood])
")

# Access the created objects from R:
prior      <- py$prior
likelihood <- py$likelihood
simulator  <- py$simulator

# (Optional) quick smoke test that the functions work:
theta_draw <- prior()
theta_draw
# $theta
# [1] 0.34  (example)

likelihood(theta_draw$theta)
# $k
# [1] 7  (example)

# draw a single sample
one <- simulator$sample(1L)
one
# e.g. $theta (1,1) array and $k (1,1) array

# draw a batch of 10,000
sim_out <- simulator$sample(10000L)
str(sim_out)

# convert to R vectors for quick use
theta <- as.numeric(sim_out$theta)  # prior draws
k     <- as.integer(sim_out$k)      # binomial outcomes
table(k)[as.character(0:10)]  # frequency table


# Setup adapter and workflow ----------------------------------------------

# To pass the output from the simulator into the networks, we use a bf.Adapter 
# to transform the simulator outputs into dictionary with keys that the approximator 
# knows how to deal with. A continuous approximator accepts the following keys:
# - "inference_variables": Variables for which to do the posterior approximation 
# (i.e., parameters) using an inference network.
# - "inference_conditions": Variables that are directly passed as conditions to 
# the inference network.
# - "summary_variables": Variables that are passed into a summary network (optional). 
# The output of the summary network is then passed together with "inference_conditions" 
# into the inference network.

# Adapter
# Here we ensure the parameter theta is constrained between 0 and 1
py_run_string("
adapter = (
    bf.Adapter()
    .constrain('theta', lower=0, upper=1)
    .rename('theta', 'inference_variables')
    .rename('k',     'inference_conditions')
)
")
adapter <- py$adapter
adapter

# Workflow
py_run_string("
inference_network = bf.networks.CouplingFlow()
workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    inference_network=inference_network
)
")
workflow <- py$workflow


# Training ----------------------------------------------------------------

# Run training
py_run_string("
history = workflow.fit_online(epochs=30)
")

# Evaluate model on test data
py_run_string("
test_data = simulator.sample(1000)
figs=workflow.plot_default_diagnostics(test_data=test_data, num_samples=500)
")

# Save figures to ./plots/
py_run_string("
import os, matplotlib
matplotlib.use('Agg')  # ensure headless save
os.makedirs('plots', exist_ok=True)

from collections.abc import Mapping

def save_all_figs(figs, prefix='bf_diag', outdir='plots'):
    paths = []
    if isinstance(figs, Mapping):          # e.g., {'losses': Figure, ...}
        for k, f in figs.items():
            p = os.path.join(outdir, f'{prefix}_{k}.png')
            f.savefig(p, dpi=200, bbox_inches='tight'); paths.append(p)
    elif isinstance(figs, (list, tuple)):  # e.g., [Figure, Figure, ...]
        for i, f in enumerate(figs, 1):
            p = os.path.join(outdir, f'{prefix}_{i:02d}.png')
            f.savefig(p, dpi=200, bbox_inches='tight'); paths.append(p)
    else:                                   # single Figure
        p = os.path.join(outdir, f'{prefix}.png')
        figs.savefig(p, dpi=200, bbox_inches='tight'); paths.append(p)
    print('Saved:', paths)
    return paths

saved_paths = save_all_figs(figs)
")

# Infer theta for observed k=7
py_run_string("
inference_data = dict(k = np.array([[7]]))
samples = workflow.sample(conditions=inference_data, num_samples=1000)
")

py_run_string('
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import beta

os.makedirs("plots", exist_ok=True)

th = samples["theta"].reshape(-1)

plt.figure()
plt.hist(th, density=True, bins=np.arange(0, 1.05, 0.05),
         color="lightgray", edgecolor="black", label="Posterior samples")

# Analytic Beta(1+7, 1+10-7) = Beta(8,4) overlay
x = np.linspace(0, 1, 200)
plt.plot(x, beta.pdf(x, a=8, b=4), ls="--", label="Analytic Beta(8,4)")

plt.xlabel("θ")
plt.ylabel("Density")
plt.title("p(θ | k=7)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/post_theta_k7.png", dpi=200, bbox_inches="tight")
print("Saved plots/post_theta_k7.png")
')

