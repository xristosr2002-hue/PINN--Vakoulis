"""Run provenance: which code and environment produced a run.

Written as provenance.json next to config.json (kept out of config.json so
ExperimentConfig.load stays a pure config round-trip). Git fields are None
when the package is not running from a git checkout -- unknown, never guessed.
"""
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import scipy
import torch

_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(args: list[str], cwd: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def collect_provenance(repo_dir: str | None = None) -> dict:
    repo_dir = repo_dir or _PACKAGE_PARENT
    # Only a directory that is itself the top of a checkout counts, so a run
    # from an unrelated parent repository is not attributed to its commit.
    top = _git(["rev-parse", "--show-toplevel"], repo_dir)
    inside = top is not None and os.path.samefile(top, repo_dir)
    commit = _git(["rev-parse", "HEAD"], repo_dir) if inside else None
    # Tracked files only: ignored run outputs and stray untracked files do not
    # change the code that ran.
    status = _git(["status", "--porcelain", "--untracked-files=no"], repo_dir) if inside else None
    return {
        "git_commit": commit,
        "git_describe": _git(["describe", "--tags", "--always", "--dirty"], repo_dir) if inside else None,
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir) if inside else None,
        "git_dirty": None if status is None else bool(status),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "command": list(sys.argv),
    }


def save_provenance(run_dir: str, repo_dir: str | None = None) -> str:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "provenance.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(collect_provenance(repo_dir), fh, indent=2)
    return path
