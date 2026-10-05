import numpy as np

from takeoff_pinn.config import DataConfig, ParamSpace, PhysicsConfig, TrueParams
from takeoff_pinn.data import make_synthetic_data
from takeoff_pinn.identifiability import (
    gauss_newton_covariance, identifiability_report, sensitivity_matrix,
)

PHYS, TRUTH = PhysicsConfig(), TrueParams()
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


def test_covariance_is_symmetric_positive_semidefinite():
    S, names = sensitivity_matrix(THETA, _t_data(), PHYS, ParamSpace(), SCALES)
    resid = np.random.default_rng(0).normal(0.0, 0.01, S.shape[0])
    cov = gauss_newton_covariance(S, resid)
    assert cov.shape == (len(names), len(names))
    assert np.allclose(cov, cov.T)
    assert (np.linalg.eigvalsh(cov) > -1e-12).all()
