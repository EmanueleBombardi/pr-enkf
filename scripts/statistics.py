"""EnKF vs PR-EnKF over 30 independent noise realisations (about 15 s).

For each measurement layout, every realisation draws new measurement noise and infers
(kappa, A+, lambda) at Re_tau = 550 with both filters. Reported per filter:
  failed runs       : a member reached non-physical parameters
  misfit to data    : RMSE between the ensemble-mean prediction and the noisy measurements
  error vs truth    : RMSE against the noise-free truth over the whole profile
  95% coverage      : fraction of the profile at Re_tau = 5200 where the truth lies
                      inside the ensemble 95% band
Averages are over successful runs only.
"""
import numpy as np
from pr_enkf import run
from pr_enkf.models.channel import truth_at, truth_profile, model_profile, make_forward

P = np.array([0.41, 26.0, 0.09])
GAMMA = np.array([0.1, 0.1, 0.1])
RE_MEAS, SIGMA_D, N_REAL = 550, 0.3, 30
LAYOUTS = {"5 sparse": np.array([5, 15, 60, 200, 450.0]),
           "8 outer": np.geomspace(30, 550, 8)}


def error_vs_truth(alpha, re_tau):
    y, U = truth_profile(re_tau)
    return np.sqrt(np.mean((model_profile(alpha, re_tau, y)[1] - U) ** 2))


def coverage(ensemble, re_tau):
    y, U = truth_profile(re_tau)
    Q = np.array([model_profile(a, re_tau, y)[1] for a in ensemble.T])
    lo, hi = np.percentile(Q, [2.5, 97.5], axis=0)
    return np.mean((U >= lo) & (U <= hi))


print(f"Literature values, error vs truth: Re550 {error_vs_truth(P, 550):.2f}, "
      f"Re180 {error_vs_truth(P, 180):.2f}, Re5200 {error_vs_truth(P, 5200):.2f}\n")
print("| Measurements | Filter | Failed runs | Misfit to data | Error, Re=550 | "
      "Error, Re=180 | Error, Re=5200 | 95% coverage, Re=5200 |")
print("|---|---|---|---|---|---|---|---|")
for layout, y_obs in LAYOUTS.items():
    forward = make_forward(RE_MEAS, y_obs)
    q_true = truth_at(RE_MEAS, y_obs)
    for name, reg in [("EnKF", False), ("PR-EnKF", True)]:
        res, n_fail = [], 0
        for k in range(N_REAL):
            d = q_true + SIGMA_D * np.random.default_rng(100 + k).standard_normal(y_obs.size)
            r = run(forward, P, GAMMA, d, SIGMA_D, n_ens=50, n_iter=30,
                    regularise=reg, inflation=1.05, seed=k, verbose=False)
            if r["failed_at"] is not None:
                n_fail += 1
                continue
            ens = r["alpha"][-1]
            m = ens.mean(axis=1)
            res.append([np.sqrt(np.mean((forward(m) - d) ** 2)), error_vs_truth(m, 550),
                        error_vs_truth(m, 180), error_vs_truth(m, 5200), coverage(ens, 5200)])
        a = np.mean(res, axis=0)
        print(f"| {layout} | {name} | {n_fail}/{N_REAL} | {a[0]:.2f} | {a[1]:.2f} | "
              f"{a[2]:.2f} | {a[3]:.2f} | {100 * a[4]:.0f}% |")
