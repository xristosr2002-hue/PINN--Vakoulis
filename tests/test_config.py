import json

from takeoff_pinn.config import (
    DataConfig, ExperimentConfig, PhysicsConfig, ParamSpace, TrainConfig,
    TrueParams, seed_everything,
)


def test_physics_derived_quantities():
    phys = PhysicsConfig()
    assert abs(phys.W - 15.0 * 9.80665) < 1e-12
    assert abs(phys.a_scale - 56.0 / 15.0) < 1e-12  # T_MAX/m, NOT rpm/60


def test_config_save_roundtrip(tmp_path):
    exp = ExperimentConfig(name="t", out_dir=str(tmp_path))
    p = exp.save()
    blob = json.loads(open(p, encoding="utf-8").read())
    assert blob["train"]["seed"] == exp.train.seed
    assert blob["physics"]["T_MAX"] == 56.0


def test_seed_everything_reproducible():
    import numpy as np
    import torch
    seed_everything(123)
    a = (np.random.rand(3), torch.rand(3))
    seed_everything(123)
    b = (np.random.rand(3), torch.rand(3))
    assert (a[0] == b[0]).all() and bool((a[1] == b[1]).all())


def test_priors_differ_from_truth():
    """Audit finding #3: priors must NOT be centered on synthetic truth."""
    truth, space = TrueParams(), ParamSpace()
    assert space.prior_mu != truth.mu
    assert space.prior_cd != truth.cd_scale
    assert space.prior_eta != truth.eta
