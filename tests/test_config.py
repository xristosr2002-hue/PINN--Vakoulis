import json

from takeoff_pinn.config import (
    DataConfig, ExperimentConfig, PhysicsConfig, ParamSpace, TrainConfig,
    TrueParams, seed_everything,
)


def test_physics_derived_quantities():
    phys = PhysicsConfig()
    assert abs(phys.W - 15.0 * 9.80665) < 1e-12
    assert abs(phys.a_scale - phys.T_MAX / 15.0) < 1e-12  # T_MAX/m, NOT rpm/60


def test_slide_specification_values():
    """Project brief (slide): S = 5 m^2, m = 15 kg, rho = 1.225, 5000 RPM, 13.5 s."""
    phys = PhysicsConfig()
    assert (phys.S, phys.m, phys.rho, phys.RPM_MAX, phys.TF) == (5.0, 15.0, 1.225, 5000.0, 13.5)


def test_config_save_roundtrip(tmp_path):
    exp = ExperimentConfig(name="t", out_dir=str(tmp_path))
    p = exp.save()
    blob = json.loads(open(p, encoding="utf-8").read())
    assert blob["train"]["seed"] == exp.train.seed
    assert blob["physics"]["T_MAX"] == exp.physics.T_MAX
    assert blob["data"]["strata_edges"] == list(exp.data.strata_edges)


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
    assert space.prior_cl != truth.cl_scale  # nominal (CFD-like) c_L is 5 % off


def test_config_load_roundtrip(tmp_path):
    exp = ExperimentConfig(name="rt", out_dir=str(tmp_path))
    exp.space.estimate_cl = True
    exp.data.truth_mu_speed_slope = 0.02
    loaded = ExperimentConfig.load(exp.save())
    assert loaded == exp
