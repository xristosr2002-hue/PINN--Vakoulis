import numpy as np
import torch

from takeoff_pinn.config import DataConfig, PhysicsConfig, TrueParams
from takeoff_pinn.data import integrate_rk4, make_synthetic_data, train_val_split

torch.set_default_dtype(torch.float64)
PHYS = PhysicsConfig()
TRUTH = TrueParams()


def test_rk4_fourth_order_convergence():
    """Halving dt must shrink the end-state error ~16x (allow >=8x).

    Grids must resolve the fastest input feature (RPM blend-to-max, ~17 ms
    timescale), so the asymptotic pair dt=0.02 vs dt=0.01 is used; coarser
    grids are pre-asymptotic for this scenario (measured ratio there ~3).
    """
    ref_t = np.linspace(0.0, PHYS.TF, 12801)
    x_ref, v_ref = integrate_rk4(ref_t, TRUTH.mu, TRUTH.cd_scale, TRUTH.cl_scale, TRUTH.eta, PHYS)
    errs = []
    for n in (401, 801):
        t = np.linspace(0.0, PHYS.TF, n)
        x, v = integrate_rk4(t, TRUTH.mu, TRUTH.cd_scale, TRUTH.cl_scale, TRUTH.eta, PHYS)
        errs.append(abs(float(v[0, -1] - v_ref[0, -1])))
    assert errs[0] / max(errs[1], 1e-15) >= 8.0


def test_rk4_batched_matches_single():
    t = np.linspace(0.0, PHYS.TF, 201)
    mu = torch.tensor([TRUTH.mu, TRUTH.mu, TRUTH.mu])
    cd = torch.tensor([TRUTH.cd_scale] * 3)
    cl = torch.tensor([TRUTH.cl_scale] * 3)
    eta = torch.tensor([TRUTH.eta] * 3)
    xb, vb = integrate_rk4(t, mu, cd, cl, eta, PHYS)
    xs, vs = integrate_rk4(t, TRUTH.mu, TRUTH.cd_scale, TRUTH.cl_scale, TRUTH.eta, PHYS)
    assert xb.shape == (3, 201)
    assert torch.allclose(xb[1], xs[0]) and torch.allclose(vb[2], vs[0])


def test_trajectory_is_physical():
    t = np.linspace(0.0, PHYS.TF, 401)
    x, v = integrate_rk4(t, TRUTH.mu, TRUTH.cd_scale, TRUTH.cl_scale, TRUTH.eta, PHYS)
    assert float(v.min()) >= -1e-9 and float(v[0, -1]) > 10.0
    assert bool((x[0, 1:] >= x[0, :-1] - 1e-12).all())


def test_synthetic_data_reproducible_and_ic_pinned():
    d1 = make_synthetic_data(PHYS, TRUTH, DataConfig(), seed=11)
    d2 = make_synthetic_data(PHYS, TRUTH, DataConfig(), seed=11)
    d3 = make_synthetic_data(PHYS, TRUTH, DataConfig(), seed=12)
    assert np.array_equal(d1["v_data"], d2["v_data"])
    assert not np.array_equal(d1["v_data"], d3["v_data"])
    assert d1["x_data"][0] == PHYS.X0 and d1["v_data"][0] == PHYS.V0
    assert len(d1["t_data"]) == len(set(d1["t_data"].tolist()))


def test_train_val_split_disjoint_and_complete():
    tr, va = train_val_split(100)
    assert set(tr).isdisjoint(va) and len(tr) + len(va) == 100
    assert 0 in tr  # keep the t=0 anchor point in training


def test_liftoff_observation_is_drawn_last_and_near_truth():
    """Adding the liftoff measurement must not change the V/x samples."""
    from takeoff_pinn.data import predict_liftoff

    on = make_synthetic_data(PHYS, TRUTH, DataConfig(observe_liftoff=True), seed=11)
    off = make_synthetic_data(PHYS, TRUTH, DataConfig(observe_liftoff=False), seed=11)
    assert np.array_equal(on["v_data"], off["v_data"]) and np.array_equal(on["x_data"], off["x_data"])
    assert off["t_liftoff_obs"] is None
    t_true = predict_liftoff(TRUTH.mu, TRUTH.cd_scale, TRUTH.cl_scale, TRUTH.eta, PHYS)["t"][0]
    assert abs(on["t_liftoff_obs"] - t_true) < 5 * DataConfig().sigma_t_liftoff


def test_linearized_liftoff_offset_matches_event_shift():
    from takeoff_pinn.data import event_nodes, fine_grid, liftoff_time_offset, predict_liftoff

    t_obs = predict_liftoff(TRUTH.mu, TRUTH.cd_scale, TRUTH.cl_scale, TRUTH.eta, PHYS)["t"][0]
    mu = TRUTH.mu * 1.03
    t_grid = fine_grid(event_nodes(t_obs), t_obs + 0.01)
    _, v = integrate_rk4(t_grid, mu, TRUTH.cd_scale, TRUTH.cl_scale, TRUTH.eta, PHYS)
    offset, _ = liftoff_time_offset(t_grid, v, TRUTH.cl_scale, t_obs, PHYS)
    exact = predict_liftoff(mu, TRUTH.cd_scale, TRUTH.cl_scale, TRUTH.eta, PHYS)["t"][0] - t_obs
    assert abs(float(offset[0]) - exact) < 0.05 * abs(exact)
