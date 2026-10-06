import os

import numpy as np
import torch

from takeoff_pinn.config import (
    DataConfig, ExperimentConfig, ParamSpace, PhysicsConfig, TrainConfig, TrueParams,
)
from takeoff_pinn.training import run_experiment

torch.set_default_dtype(torch.float64)


def _quick_cfg(tmp_path) -> ExperimentConfig:
    return ExperimentConfig(
        name="smoke",
        out_dir=str(tmp_path),
        data=DataConfig(n_data=60),
        train=TrainConfig(
            seed=7, n_restarts=1, pretrain_epochs=40, epochs_adam=60,
            ramp_center=20.0, ramp_width=8.0, n_collocation=32,
            resample_epochs=(40,), n_resample_add=8,
            use_lbfgs=True, lbfgs_max_iter=15, val_every=10, oe_max_nfev=60,
        ),
    )


def test_run_experiment_smoke(tmp_path):
    results = run_experiment(_quick_cfg(tmp_path))
    m = results["metrics"]
    sp = ParamSpace()
    assert np.isfinite(m["v_rmse_data"]) and np.isfinite(m["r_dyn_rms"])
    assert sp.mu_bounds[0] <= m["mu"] <= sp.mu_bounds[1]
    assert sp.eta_bounds[0] <= m["eta"] <= sp.eta_bounds[1]
    # Honest pure-ML errors on a single noisy run: bounded by the mu-eta valley
    # spread (no prior is pinning these; see AUDIT_REPORT 2.3.2/2.3.3).
    assert m["mu_abs_err"] < 0.06
    assert m["eta_abs_err"] < 0.15
    # Mechanics check only: 100 epochs cannot fit the 13.5 s record (V-RMSE
    # ~1.2 m/s here; the full preset reaches 0.09 = noise level, see E04).
    # Fit quality is judged by the multi-seed experiments, not this test.
    assert m["v_rmse_data"] < 2.0
    # artifacts on disk
    for fname in ("config.json", "metrics.json", "model.pt", "history.json"):
        assert os.path.exists(os.path.join(str(tmp_path), "smoke", fname)), fname
    assert results["identifiability"]["max_corr"] <= 1.0
    # Uncertainties are first-class outputs (estimates without error bars are
    # meaningless for this weakly-identified problem). v2: honest per-channel
    # sandwich sigmas replace the v1 '*_gn_sigma' keys (which were 1.7x low).
    for key in ("mu_sigma", "cd_scale_sigma", "eta_sigma", "kappa_sigma"):
        assert key in m and m[key] > 0.0
    # classical output-error baseline reported alongside the PINN
    for key in ("mu_oe", "cd_scale_oe", "eta_oe", "kappa_oe", "x_liftoff_oe"):
        assert key in m and np.isfinite(m[key])


def test_run_experiment_with_observed_liftoff_estimates_cl(tmp_path):
    """C9 path end to end: weighted OE + liftoff row + PINN event term."""
    exp = _quick_cfg(tmp_path)
    exp.name = "smoke_c9"
    exp.train.oe_weighted = True
    exp.space.estimate_cl = True
    exp.space.prior_cl = 1.05
    m = run_experiment(exp)["metrics"]
    assert m["liftoff_observed"] is True
    assert "cl_scale_sigma" in m and 0.0 < m["cl_scale_sigma"] < 0.05
    assert abs(m["cl_scale_oe"] - 1.0) < 0.03  # OE recovers c_L from the 1.05 start
    assert np.isfinite(m["gen_ground_roll_pct_err"])
