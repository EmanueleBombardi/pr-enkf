import numpy as np
from pr_enkf import analysis, run, ForwardModelFailure
from pr_enkf.models.channel import model_profile, truth_profile


def linear_gaussian_setup(n_ens, seed=0):
    """Linear model q = H alpha, Gaussian ensemble and measurement noise."""
    rng = np.random.default_rng(seed)
    H = np.array([[1.0, 0.5], [0.2, 1.0], [0.7, -0.3]])
    p, std = np.array([2.0, 3.0]), np.array([0.2, 0.3])
    alpha = p[:, None] + std[:, None] * rng.standard_normal((2, n_ens))
    sigma_d = 0.1
    d = H @ np.array([2.3, 2.6])
    d_pert = d[:, None] + sigma_d * rng.standard_normal((3, n_ens))
    return H, p, std, alpha, sigma_d, d, d_pert


def test_enkf_matches_kalman_filter_for_linear_gaussian():
    """One EnKF step on a linear-Gaussian problem converges to the exact Kalman update."""
    H, p, std, alpha, sigma_d, d, d_pert = linear_gaussian_setup(n_ens=20000)
    alpha_a = analysis(alpha, H @ alpha, d_pert, sigma_d, eps_rel=0.0)
    P, C_dd = np.diag(std**2), sigma_d**2 * np.eye(3)
    exact = p + P @ H.T @ np.linalg.solve(H @ P @ H.T + C_dd, d - H @ p)
    np.testing.assert_allclose(alpha_a.mean(axis=1), exact, atol=0.01)


def test_pr_enkf_tends_to_enkf_for_large_cpp():
    """C_pp -> infinity recovers the standard EnKF."""
    H, p, std, alpha, sigma_d, d, d_pert = linear_gaussian_setup(n_ens=200)
    p_pert = p[:, None] * np.ones_like(alpha)
    a_enkf = analysis(alpha, H @ alpha, d_pert, sigma_d, eps_rel=0.0)
    a_pr = analysis(alpha, H @ alpha, d_pert, sigma_d, eps_rel=0.0,
                    p_pert=p_pert, gamma=np.array([1e4, 1e4]))
    np.testing.assert_allclose(a_pr, a_enkf, rtol=1e-4)


def test_pr_enkf_pulls_towards_p_when_data_are_uninformative():
    H, p, std, alpha, sigma_d, d, d_pert = linear_gaussian_setup(n_ens=200)
    shifted = alpha + 1.0                       # ensemble away from p
    p_pert = p[:, None] * np.ones_like(alpha)
    a = analysis(shifted, H @ shifted, d_pert, sigma_d=1e4,
                 p_pert=p_pert, gamma=np.array([0.01, 0.01]))
    assert np.all(np.abs(a.mean(axis=1) - p) < np.abs(shifted.mean(axis=1) - p))


def test_inflation_scales_spread_and_keeps_mean():
    H, p, std, alpha, sigma_d, d, d_pert = linear_gaussian_setup(n_ens=100)
    a1 = analysis(alpha, H @ alpha, d_pert, sigma_d)
    a2 = analysis(alpha, H @ alpha, d_pert, sigma_d, inflation=1.05)
    np.testing.assert_allclose(a2.std(axis=1), 1.05 * a1.std(axis=1))
    np.testing.assert_allclose(a2.mean(axis=1), a1.mean(axis=1))


def test_run_stops_cleanly_on_forward_model_failure():
    def forward(alpha):
        raise ForwardModelFailure("always fails")
    r = run(forward, [1.0], [0.1], [1.0], 0.1, n_ens=10, n_iter=5, verbose=False)
    assert r["failed_at"] == 0


def test_channel_viscous_sublayer():
    """U+ = y+ close to the wall, for the model and the truth."""
    for y, U in (model_profile((0.41, 26.0, 0.09), 550), truth_profile(550)):
        near = y < 1.0
        np.testing.assert_allclose(U[near], y[near], rtol=0.01, atol=1e-3)
