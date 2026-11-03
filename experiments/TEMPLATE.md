# E<NN> — <short title>

Copy to `experiments/E<NN>_<slug>.md` (or add an entry to `LOG.md`). Fill every field; write
"n/a" rather than leaving one out. Commit the record and its small JSON outputs, never `results/`.

| Field | Value |
|---|---|
| Experiment ID | E<NN> (change C<NN> if it motivates a code change) |
| Branch / git commit | `experiments/<name>` @ `<hash>` (from the runs' `provenance.json`; must not be dirty) |
| Date | YYYY-MM-DD |
| Purpose / hypothesis | |
| Command | `python scripts/multiseed.py --tag E<NN>_<slug> --seeds 7 17 27 37 47 57 67 77 87 97 --jobs 4 --set ...` |
| Config overrides vs defaults | `--set` list (full config in each run's `config.json`) |
| Seeds | |
| Environment | from `provenance.json` (Python, torch, numpy, scipy, threads) |
| Artifacts | `experiments/E<NN>_<slug>/{multiseed,config,metrics}.json` |

## Setup

- **Physics / physical parameters:** (changes vs `PhysicsConfig`, `TrueParams`)
- **Initial conditions / liftoff observation:**
- **Data:** n_data, σ_V, σ_x, sampling strata
- **Model architecture:**
- **Optimizer / training:** epochs, learning rates, L-BFGS, restarts
- **Loss weights / curriculum:**
- **Collocation sampling:**

## Results (vs truth and vs the reference experiment)

| Metric | Reference | This experiment | Change |
|---|---|---|---|
| Parameter errors (μ, c_D, c_L, η, κ) and σ coverage | | | |
| Liftoff time / ground roll error | | | |
| State RMSE (V, x) vs truth | | | |
| Final loss components (data, dyn, kin) | | | |
| Runtime | | | |

## Log entry

- **CHANGE:**
- **WHY:**
- **EXPECTED:**
- **ACTUAL:**
- **DECISION:** keep / reject / opt-in only
