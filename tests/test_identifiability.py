import numpy as np

from takeoff_pinn.config import DataConfig, ParamSpace, PhysicsConfig, TrueParams
from takeoff_pinn.data import make_synthetic_data
from takeoff_pinn.identifiability import (
    gauss_newton_covariance, identifiability_report, sensitivity_matrix,
)

# The facts below were established for the v1 scenario (S = 0.55 m^2, 8 s,
# 56 N), where W - L stays >> NORMAL_SOFTNESS. They are pinned to that
# configuration explicitly; slide-scenario identifiability is tested separately.
PHYS, TRUTH = PhysicsConfig(S=0.55, TF=8.0, T_MAX=56.0), TrueParams()
THETA = {"mu": 0.08, "cd_scale": 1.15, "cl_scale": 1.0, "eta": 0.91}
SCALES = {"v": 14.0, "x": 70.0}


def _t_data():
    return make_synthetic_data(PHYS, TRUTH, DataConfig(n_data=60), seed=3)["t_data"]


def test_estimating_cl_is_structurally_degenerate():
    """Audit finding #2: cd/cl sensitivity columns are ~exactly anti-collinear."""
    S, names = sensitivity_matrix(THETA, _t_data(), PHYS, ParamSpace(estimate_cl=True), SCALES)
    rep = identifiability_report(S, names)
    i, j = names.index("cd_scale"), names.index("cl_scale")
    assert abs(rep["corr"][i][j]) > 0.999


def test_lumped_coefficient_kappa_is_the_degenerate_invariant():
    """While N > 0 the quadratic-in-q resistance is qS*(CD0*cd - mu*CL0*cl):
    lift *reduces* friction, so raising cl must be offset by raising cd.
    Moving (cd, cl) at constant kappa leaves the trajectory unchanged; the
    sign-flipped combination (as misprinted in the v1 audit/thesis) does not."""
    from takeoff_pinn.data import integrate_rk4
    from takeoff_pinn.identifiability import lumped_drag_coefficient

    t = np.linspace(0.0, PHYS.TF, 401)
    mu, cl_new = 0.08, 1.2
    kappa0 = lumped_drag_coefficient(1.15, 1.0, mu, PHYS)
    cd_new = (kappa0 + mu * PHYS.CL0 * cl_new) / PHYS.CD0
    assert abs(lumped_drag_coefficient(cd_new, cl_new, mu, PHYS) - kappa0) < 1e-15

    x0, _ = integrate_rk4(t, mu, 1.15, 1.0, 0.91, PHYS)
    x1, _ = integrate_rk4(t, mu, cd_new, cl_new, 0.91, PHYS)
    cd_wrong = (PHYS.CD0 * 1.15 + mu * PHYS.CL0 * (1.0 - cl_new)) / PHYS.CD0
    x2, _ = integrate_rk4(t, mu, cd_wrong, cl_new, 0.91, PHYS)
    assert float((x1 - x0).abs().max()) < 1e-3
    assert float((x2 - x0).abs().max()) > 0.1


def test_fixed_cl_removes_exact_degeneracy():
    """3-param set is invertible (cond ~4e2); 4-param set is rank-deficient (~7e5)."""
    t = _t_data()
    S3, names3 = sensitivity_matrix(THETA, t, PHYS, ParamSpace(estimate_cl=False), SCALES)
    S4, _ = sensitivity_matrix(THETA, t, PHYS, ParamSpace(estimate_cl=True), SCALES)
    rep3 = identifiability_report(S3, names3)
    rep4 = identifiability_report(S4, ["mu", "cd_scale", "cl_scale", "eta"])
    assert "cl_scale" not in names3
    assert rep3["condition_number"] < 2.0e3
    assert rep4["condition_number"] > 100.0 * rep3["condition_number"]


def test_mu_eta_near_collinearity_is_exposed():
    """Known physics of a single ground roll: after spool-up both eta*T(t) and
    mu*N are near-constant forces, so their trajectory sensitivities are highly
    correlated (~ -0.9999 here). The report must expose this so the correlated
    UQ (not independent sigmas) handles it. Separation comes only from the
    spool-up/excitation window."""
    S, names = sensitivity_matrix(THETA, _t_data(), PHYS, ParamSpace(estimate_cl=False), SCALES)
    rep = identifiability_report(S, names)
    i, j = names.index("mu"), names.index("eta")
    assert abs(rep["corr"][i][j]) > 0.99


def _two_channel_problem(rng, n=150, sig=(0.005, 0.001)):
    S = rng.normal(size=(2 * n, 3)) * np.array([1.0, 0.5, 2.0])
    noise_sd = np.r_[np.full(n, sig[0]), np.full(n, sig[1])]
    return S, noise_sd


def test_parameter_covariance_matches_empirical_scatter_heteroscedastic():
    """Unweighted LS on two channels with 5x different noise: the sandwich
    covariance must reproduce the Monte-Carlo scatter of the estimates (the
    pooled homoscedastic formula does not)."""
    from takeoff_pinn.identifiability import parameter_covariance

    rng = np.random.default_rng(1)
    S, sd = _two_channel_problem(rng)
    est, reported = [], []
    for _ in range(3000):
        e = rng.normal(size=sd.shape) * sd
        beta = np.linalg.lstsq(S, e, rcond=None)[0]
        est.append(beta)
        reported.append(np.diag(parameter_covariance(S, e - S @ beta)))
    empirical = np.var(np.array(est), axis=0)
    ratio = np.mean(reported, axis=0) / empirical
    assert np.all(np.abs(ratio - 1.0) < 0.12)


def test_parameter_covariance_weighted_reduces_to_inverse_fisher():
    from takeoff_pinn.identifiability import parameter_covariance

    rng = np.random.default_rng(2)
    S, sd = _two_channel_problem(rng)
    r = rng.normal(size=sd.shape) * sd
    n, p = S.shape
    var = np.r_[np.full(n // 2, np.mean(r[: n // 2] ** 2)),
                np.full(n // 2, np.mean(r[n // 2:] ** 2))] * n / (n - p)
    cov = parameter_covariance(S, r, weights=1.0 / var)
    assert np.allclose(cov, np.linalg.inv((S.T / var) @ S), rtol=1e-10)


def test_parameter_covariance_rejects_misaligned_residual():
    """E00 bug: S built at all data times, residual from the training subset."""
    import pytest
    from takeoff_pinn.identifiability import parameter_covariance

    S = np.ones((20, 3))
    with pytest.raises(ValueError):
        parameter_covariance(S, np.zeros(16))


def test_covariance_is_symmetric_positive_semidefinite():
    S, names = sensitivity_matrix(THETA, _t_data(), PHYS, ParamSpace(), SCALES)
    resid = np.random.default_rng(0).normal(0.0, 0.01, S.shape[0])
    cov = gauss_newton_covariance(S, resid)
    assert cov.shape == (len(names), len(names))
    assert np.allclose(cov, cov.T)
    assert (np.linalg.eigvalsh(cov) > -1e-12).all()


# ---- slide scenario (S = 5 m^2, TF = 13.5 s) -----------------------------

SLIDE = PhysicsConfig()


def test_slide_kappa_invariant_holds_until_liftoff():
    """Same kappa => same trajectory while both vehicles are on the ground.
    (A finite cl change moves the liftoff event itself, which is physics.)
    With the v1 5 N softplus this held only to ~4e-3 m; with 0.5 N to ~3e-4 m."""
    from takeoff_pinn.data import integrate_rk4
    from takeoff_pinn.evaluate import predict_liftoff
    from takeoff_pinn.identifiability import lumped_drag_coefficient

    mu, cl_new = 0.08, 1.2
    kappa0 = lumped_drag_coefficient(1.15, 1.0, mu, SLIDE)
    cd_new = (kappa0 + mu * SLIDE.CL0 * cl_new) / SLIDE.CD0
    t_end = predict_liftoff(mu, cd_new, cl_new, 0.91, SLIDE)["t"][0]  # earlier liftoff
    t = np.linspace(0.0, t_end, 1001)
    x0, _ = integrate_rk4(t, mu, 1.15, 1.0, 0.91, SLIDE)
    x1, _ = integrate_rk4(t, mu, cd_new, cl_new, 0.91, SLIDE)
    assert float((x1 - x0).abs().max()) < 1e-3


def test_slide_lift_unloading_decorrelates_mu_eta_sensitivities():
    """At S = 5 m^2 lift unloads the wheels during the run, so mu*N varies in
    time while eta*T is ~flat: the *sensitivity-column* correlation of (mu, eta)
    drops from -0.9999 (v1) to ~-0.87. (The parameter *estimates* of mu, c_D,
    eta stay strongly correlated along a joint thrust-vs-resistance valley.)"""
    t = np.linspace(0.0, SLIDE.TF, 160)
    S, names = sensitivity_matrix(THETA, t, SLIDE, ParamSpace(), {"v": 13.0, "x": 113.0})
    rep = identifiability_report(S, names)
    assert abs(rep["corr"][names.index("mu")][names.index("eta")]) < 0.95


def test_slide_cd_cl_still_degenerate_in_linearization():
    t = np.linspace(0.0, SLIDE.TF, 160)
    S, names = sensitivity_matrix(THETA, t, SLIDE, ParamSpace(estimate_cl=True),
                                  {"v": 13.0, "x": 113.0})
    rep = identifiability_report(S, names)
    assert abs(rep["corr"][names.index("cd_scale")][names.index("cl_scale")]) > 0.9999


def test_liftoff_row_resolves_cd_cl_degeneracy():
    """Linearized Fisher: adding the observed liftoff time makes (c_D, c_L)
    separately identifiable (sigma(c_L) collapses from O(1) to O(1e-3))."""
    from takeoff_pinn.data import predict_liftoff
    from takeoff_pinn.identifiability import event_sensitivity, parameter_covariance

    space = ParamSpace(estimate_cl=True)
    t = np.linspace(0.0, SLIDE.TF, 160)
    S, names = sensitivity_matrix(THETA, t, SLIDE, space, {"v": 13.0, "x": 113.0})
    r = np.r_[np.full(160, 0.10 / 13.0), np.full(160, 0.08 / 113.0)]  # |r| = noise level
    w = 1.0 / r**2
    t_obs = predict_liftoff(THETA["mu"], THETA["cd_scale"], THETA["cl_scale"], THETA["eta"], SLIDE)["t"][0]
    K = event_sensitivity(THETA, {"t": t_obs, "sigma": 0.02}, SLIDE, space)
    j = names.index("cl_scale")
    sd_without = np.sqrt(parameter_covariance(S, r, weights=w)[j, j])
    sd_with = np.sqrt(parameter_covariance(S, r, weights=w, S_known=K)[j, j])
    assert sd_without > 0.1 and sd_with < 0.01
