import os

from takeoff_pinn.run import build_preset, main


def test_presets_exist_and_differ():
    quick, full = build_preset("quick"), build_preset("full")
    assert quick.train.epochs_adam < full.train.epochs_adam
    assert full.train.n_restarts >= 2


def test_cli_quick_run_writes_figures(tmp_path):
    rc = main(["--preset", "quick", "--name", "cli_smoke", "--out-dir", str(tmp_path)])
    assert rc == 0
    run_dir = os.path.join(str(tmp_path), "cli_smoke")
    for fname in ("states.png", "residuals.png", "parameters.png", "uq.png",
                  "metrics.json", "config.json", "provenance.json"):
        assert os.path.exists(os.path.join(run_dir, fname)), fname
