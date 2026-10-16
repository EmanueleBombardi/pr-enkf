"""
Fully developed turbulent channel flow in wall units, closed with a mixing-length model.

    (1 + nu_t+) dU+/dy+ = 1 - y+/Re_tau,      nu_t+ = (l+)^2 |dU+/dy+|

Model:  l+ = min(kappa y+, lambda Re_tau) (1 - exp(-y+/A+))
        parameters alpha = (kappa, A+, lambda), literature values p = (0.41, 26, 0.09)
Truth:  Nikuradse mixing length with van Driest damping (A+ = 26), used to generate
        synthetic measurements. The model's piecewise outer cap differs from Nikuradse's
        smooth profile, so there is a small model-form error: no parameter set
        reproduces the truth at every Re_tau.
"""
import numpy as np
from pr_enkf.filter import ForwardModelFailure


def grid(re_tau, n=400):
    """Wall-clustered grid from the wall (y+ = 0) to the centreline (y+ = Re_tau)."""
    s = np.linspace(0, 1, n)
    return re_tau * (1 - np.cos(0.5 * np.pi * s))


def solve(l_plus, y_plus, re_tau):
    """Velocity profile U+(y+) for a given mixing-length profile l+(y+).

    The momentum balance is a quadratic in g = dU+/dy+:  (l+)^2 g^2 + g - tau = 0,
    with tau = 1 - y+/Re_tau. Its positive root is integrated with the trapezoidal rule.
    """
    tau = 1 - y_plus / re_tau
    g = 2 * tau / (1 + np.sqrt(1 + 4 * l_plus**2 * tau))   # cancellation-free root
    return np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(y_plus))])


def model_profile(alpha, re_tau, y_plus=None):
    """Model velocity profile for parameters alpha = (kappa, A+, lambda).

    Raises ForwardModelFailure if any parameter is non-positive.
    """
    kappa, A, lam = alpha
    if kappa <= 0 or A <= 0 or lam <= 0:
        raise ForwardModelFailure(
            f"non-physical parameters kappa={kappa:.3f}, A+={A:.2f}, lambda={lam:.3f}")
    y = grid(re_tau) if y_plus is None else y_plus
    l = np.minimum(kappa * y, lam * re_tau) * (1 - np.exp(-y / A))
    return y, solve(l, y, re_tau)


def truth_profile(re_tau, y_plus=None):
    """Synthetic truth: Nikuradse mixing length with van Driest damping."""
    y = grid(re_tau) if y_plus is None else y_plus
    eta = y / re_tau
    l = re_tau * (0.14 - 0.08 * (1 - eta)**2 - 0.06 * (1 - eta)**4) * (1 - np.exp(-y / 26.0))
    return y, solve(l, y, re_tau)


def truth_at(re_tau, y_obs):
    """Noise-free truth at the measurement locations (solved on the full grid)."""
    y, U = truth_profile(re_tau)
    return np.interp(y_obs, y, U)


def make_forward(re_tau, y_obs):
    """Forward model alpha -> q: model U+ at the measurement locations y_obs."""
    y = grid(re_tau)

    def forward(alpha):
        _, U = model_profile(alpha, re_tau, y)
        return np.interp(y_obs, y, U)
    return forward
