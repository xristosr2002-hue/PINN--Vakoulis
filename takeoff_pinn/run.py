"""CLI: python -m takeoff_pinn.run --preset quick|full [--name N] [--out-dir D] [--csv F]"""
import argparse
import json

from .config import DataConfig, ExperimentConfig, TrainConfig
from .plots import save_all
from .training import run_experiment


def build_preset(preset: str) -> ExperimentConfig:
    if preset == "quick":
        return ExperimentConfig(
            name="quick",
            data=DataConfig(n_data=100),
            train=TrainConfig(
                n_restarts=1, pretrain_epochs=150, epochs_adam=300,
                ramp_center=80.0, ramp_width=30.0, n_collocation=64,
                resample_epochs=(180,), n_resample_add=16,
                lbfgs_max_iter=40, val_every=20, oe_max_nfev=120,
            ),
        )
    if preset == "full":
        return ExperimentConfig(name="full")  # all defaults = full settings
    raise ValueError(f"unknown preset: {preset}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Takeoff-PINN experiment runner")
    ap.add_argument("--preset", choices=("quick", "full"), default="quick")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--csv", default=None, help="optional CSV with columns t,x,V")
    ap.add_argument("--estimate-cl", action="store_true",
                    help="degeneracy ablation: also estimate cl_scale")
    args = ap.parse_args(argv)

    exp = build_preset(args.preset)
    if args.name:
        exp.name = args.name
    if args.out_dir:
        exp.out_dir = args.out_dir
    if args.csv:
        exp.data_csv = args.csv
    exp.space.estimate_cl = bool(args.estimate_cl)

    results = run_experiment(exp)
    save_all(results["fit"], results["history"], results["data"], results["uq"],
             results["run_dir"], exp.physics.V_LIFTOFF)

    print(json.dumps({k: v for k, v in results["metrics"].items()
                      if isinstance(v, (int, float))}, indent=2))
    print(f"artifacts: {results['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
