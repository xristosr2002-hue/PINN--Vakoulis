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
    # state fit should at least track the data noise level after a short run
    assert m["v_rmse_data"] < 0.5
    # artifacts on disk
    for fname in ("config.json", "metrics.json", "model.pt", "history.json"):
        assert os.path.exists(os.path.join(str(tmp_path), "smoke", fname)), fname
    assert results["identifiability"]["max_corr"] <= 1.0
    # Uncertainties are first-class outputs (estimates without error bars are
    # meaningless for this weakly-identified problem). v2: honest per-channel
    # sandwich sigmas replace the v1 '*_gn_sigma' keys (which were 1.7x low).
    for key in ("mu_sigma", "cd_scale_sigma", "eta_sigma", "kappa_sigma"):
        assert key in m and m[key] > 0.0
