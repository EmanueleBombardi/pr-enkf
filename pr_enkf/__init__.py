"""PR-EnKF: Parameter-Regularised Ensemble Kalman Filter (didactic Python implementation)."""
from pr_enkf.filter import analysis, run, ForwardModelFailure

__all__ = ["analysis", "run", "ForwardModelFailure"]
__version__ = "1.0.0"
