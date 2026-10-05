"""Monte-Carlo check of the output-error estimator and its reported uncertainty.

For each seed: fresh synthetic noise -> train split -> OE fit -> sensitivity
matrix at the training times -> covariance variants. Summarizes, per parameter,
the empirical RMSE of the estimates and, per covariance variant, whether the
reported sigma is honest (std of z = err/sigma ~ 1, 68/95 % coverage).

    python scripts/oe_montecarlo.py --seeds 40 --out experiments/E03_covariance/oe_mc.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from takeoff_pinn.config import ExperimentConfig  # noqa: E402
from takeoff_pinn.data import make_synthetic_data, train_val_split  # noqa: E402
from takeoff_pinn.identifiability import (gauss_newton_covariance,  # noqa: E402
                                          parameter_covariance, sensitivity_matrix)
from takeoff_pinn.oe import output_error_fit  # noqa: E402


def one_seed(seed: int, exp: ExperimentConfig) -> dict:
    d = make_synthetic_data(exp.physics, exp.truth, exp.data, seed)
    tr, _ = train_val_split(len(d["t_data"]))
    theta, info = output_error_fit(d["t_data"][tr], d["x_data"][tr], d["v_data"][tr],
                                   exp.physics, exp.space, exp.train)
    S, names = sensitivity_matrix(theta, info["t"], exp.physics, exp.space, info["scales"])
    r = np.asarray(info["residual_data"])
    S_all, _ = sensitivity_matrix(theta, d["t_data"], exp.physics, exp.space, info["scales"])
    covs = {
        "v1_code": gauss_newton_covariance(S_all, r),     # E00 behaviour (inconsistent)
        "pooled": gauss_newton_covariance(S, r),          # consistent, homoscedastic
        "reported": parameter_covariance(S, r, weights=info.get("weights")),
    }
    truth = {n: getattr(exp.truth, n) for n in names}
    return {
        "seed": seed, "names": names,
        "err": [theta[n] - truth[n] for n in names],
        "sigma": {k: np.sqrt(np.clip(np.diag(c), 0, None)).tolist() for k, c in covs.items()},
    }


def summarize(rows: list[dict]) -> dict:
    names = rows[0]["names"]
    err = np.array([r["err"] for r in rows])
    out = {"n_seeds": len(rows), "names": names,
           "rmse": dict(zip(names, np.sqrt((err**2).mean(0)).tolist())),
           "bias": dict(zip(names, err.mean(0).tolist())), "variants": {}}
    for key in rows[0]["sigma"]:
        sig = np.array([r["sigma"][key] for r in rows])
        z = err / sig
        out["variants"][key] = {
            "mean_sigma": dict(zip(names, sig.mean(0).tolist())),
            "z_std": dict(zip(names, z.std(0).tolist())),
            "cover68": dict(zip(names, (np.abs(z) < 1.0).mean(0).tolist())),
            "cover95": dict(zip(names, (np.abs(z) < 1.96).mean(0).tolist())),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--first-seed", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    torch.set_default_dtype(torch.float64)
    exp = ExperimentConfig()
    t0 = time.time()
    rows = [one_seed(s, exp) for s in range(args.first_seed, args.first_seed + args.seeds)]
    summary = summarize(rows)
    summary["runtime_s"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
