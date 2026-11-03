import json
import os
import re
import subprocess

import pytest

from takeoff_pinn.provenance import collect_provenance, save_provenance

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _in_git_checkout() -> bool:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=10)
    except OSError:
        return False
    return out.returncode == 0


def test_environment_fields_always_present(tmp_path):
    # A directory that is not a git checkout: code identity is unknown (None),
    # never guessed, and the environment is still recorded.
    prov = collect_provenance(repo_dir=str(tmp_path))
    assert prov["git_commit"] is None and prov["git_dirty"] is None
    for key in ("python", "torch", "numpy", "scipy", "platform", "timestamp_utc", "command"):
        assert prov[key], key


@pytest.mark.skipif(not _in_git_checkout(), reason="not running from a git checkout")
def test_commit_matches_checkout():
    prov = collect_provenance()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{40}", prov["git_commit"])
    assert prov["git_commit"] == head
    assert isinstance(prov["git_dirty"], bool)


def test_save_writes_json_next_to_run(tmp_path):
    path = save_provenance(str(tmp_path / "run"))
    assert os.path.basename(path) == "provenance.json"
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    assert "git_commit" in blob and "torch" in blob
