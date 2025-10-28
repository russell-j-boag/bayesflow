# --- In a fresh R session ---
options(reticulate.use_uv = FALSE)
Sys.setenv(RETICULATE_PYTHON = "/Users/rjb779/Library/r-miniconda-arm64/envs/r-bf/bin/python")
library(reticulate)

# If macOS backend isn’t available, use a headless one and save figure.
py_run_string("
import numpy as np
import matplotlib
try:
    import matplotlib.pyplot as plt
    _SHOW = True
except Exception:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _SHOW = False

from scipy.stats import beta

# sample from the joint model
prior = np.random.beta(a=1, b=1, size=10_000)
prior_predictives = np.random.binomial(n=10, p=prior)

# rejection sampling
observed = 7
posterior = prior[prior_predictives == observed]

# plot results
x = np.linspace(0, 1, 51)
plt.figure()
plt.plot(x, beta.pdf(x, a=1, b=1),                 label='Prior (analytic)',     c='black', ls='--')
plt.plot(x, beta.pdf(x, a=1+observed, b=1+10-observed), label='Posterior (analytic)',  c='black', ls='-.')
bins = np.linspace(0, 1, 21)
plt.hist(prior,     density=True, alpha=0.5, bins=bins, label='Prior samples')
plt.hist(posterior, density=True, alpha=0.5, bins=bins, label='Posterior samples')
plt.xlabel(r'$\\theta$')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()

if _SHOW:
    plt.show()
else:
    plt.savefig('beta_rejection.png', dpi=200)
    print('Saved figure: beta_rejection.png')
")
