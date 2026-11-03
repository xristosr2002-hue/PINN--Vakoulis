# Development workflow

## Branches

| Branch | Purpose | Rule |
|---|---|---|
| `main` | Stable, validated versions | Never developed on directly. Updated only by merging `development` after the full test suite passes and the change log is updated. Every update gets a release tag. |
| `development` | Integration branch | Receives reviewed, tested work from feature/experiment branches. |
| `experiments/<name>` | One experiment or method change each (e.g. `experiments/adaptive-loss`, `experiments/fourier-features`, `experiments/rad`, `experiments/inverse-identification`, `experiments/uncertainty-quantification`) | Branched from `development`; created when the work starts, not in advance. Merged back only if the logged evidence supports it; rejected experiments stay on their branch as a record. |

## Release tags (semantic versioning)

- `MAJOR` — scenario/model changes that invalidate comparisons with earlier results (e.g. v1 → v2).
- `MINOR` — new validated capability or changed default.
- `PATCH` — bug fixes that do not change the logged conclusions.

Existing: `v1.0.0` (audited v1 package, baseline E00), `v2.0.0` (slide scenario, E00–E10).

## Per change

1. **Before:** `git status` is clean; on the right branch (`git switch -c experiments/<name> development`).
2. **Develop:** one change at a time; no unrelated edits.
3. **Validate:** `python -m pytest` (full suite before merging to `development`); multi-seed evidence
   for anything that claims an improvement; inspect `git diff`.
4. **Record:** an `experiments/LOG.md` entry (CHANGE / WHY / EXPECTED / ACTUAL / DECISION) using
   `experiments/TEMPLATE.md`; commit the small JSON outputs under `experiments/<ID>/`.
5. **Commit** with a conventional message (`feat:`, `fix:`, `docs:`, `perf:`, `test:`, `chore:`).
6. **Push** the branch: `git push -u origin <branch>`.

## Never

Force-push, `git reset --hard`, rewriting published history, committing `results/`, checkpoints,
datasets, credentials or thesis sources, or merging an experiment into `main` directly.
