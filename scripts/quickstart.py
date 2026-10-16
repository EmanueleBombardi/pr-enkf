"""Minimal example: infer (kappa, A+, lambda) from 8 noisy measurements with both filters."""
import numpy as np
from pr_enkf import run
from pr_enkf.models.channel import truth_at, make_forward

p = np.array([0.41, 26.0, 0.09])      # literature-consistent values
gamma = np.array([0.1, 0.1, 0.1])     # relative uncertainty
sigma_d = 0.3                         # measurement noise std

y_obs = np.geomspace(30, 550, 8)      # measurement locations (y+) at Re_tau = 550
d = truth_at(550, y_obs) + sigma_d * np.random.default_rng(1).standard_normal(y_obs.size)
forward = make_forward(550, y_obs)    # alpha -> q

for name, reg in [("EnKF", False), ("PR-EnKF", True)]:
    r = run(forward, p, gamma, d, sigma_d, n_ens=50, n_iter=30, regularise=reg)
    a = r["alpha"][-1]
    print(f"{name:8s} kappa = {a[0].mean():.3f} ± {a[0].std():.3f}   "
          f"A+ = {a[1].mean():.1f} ± {a[1].std():.1f}   "
          f"lambda = {a[2].mean():.3f} ± {a[2].std():.3f}")
