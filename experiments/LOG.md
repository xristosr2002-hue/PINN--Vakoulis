# Change & experiment log (takeoff_pinn_v2)

Every change is compared with E00 (`E00_baseline.md`, tag `baseline-v1`).
Format per Rule 6: CHANGE / WHY / EXPECTED EFFECT / ACTUAL EFFECT / DECISION.

---

## C1 — Correct κ and report it as a metric

- **Requested by:** Agent 2 (physics). Verified by Agents 1 and 3. Agent 4 confirmed no numeric effect on training.
- **CHANGE:** Added `identifiability.lumped_drag_coefficient` (κ = C_D0·c_D − μ·C_L0·c_L) and the
  metrics `kappa`, `kappa_true` and `kappa_rel_err`. Added a regression test that pins the invariant direction.
- **WHY:** While N > 0, the resistance is μ·W·cosγ + q·S·(C_D0·c_D − μ·C_L0·c_L). Lift *reduces*
  friction, so the sign is "−". The v1 AUDIT_REPORT and the thesis print "+", and their
  "κ matched truth to 0.12 %" check used the wrong formula. With c_L fixed, the reported c_D is
  conditional on the c_L assumption; κ is not.
- **EXPECTED:** No change to training or estimates. Correct identifiable quantity is reported.
- **ACTUAL:** Moving (c_D, c_L) at constant κ(−) changes x(t) by < 1e-3 m (3e-5 m measured).
  The "+" combination changes it by 0.36 m. New tests pass (41/41 in the full suite at C2).
- **DECISION:** Keep. Thesis and audit text are corrected in the v2 thesis sources (see T1).

---

## C2 / E02 — Remove per-call overhead in RK4, OE Jacobian and sensitivities (Agent 1)

- **CHANGE:** Exogenous inputs (rpm, headwind, slope) are evaluated once per integration grid
  (`physics.exogenous_inputs` + `accel_from_inputs`; `rhs_accel` is unchanged for callers).
  The OE fit gets scipy's own 2-point Jacobian rule through one batched RK4 call, and
  `sensitivity_matrix` batches its 2p perturbations.
- **WHY:** Profiling showed integration cost is torch call overhead (0.77 s / 530 steps; B=8 costs the
  same as B=1). The OE stage took 28 s per restart, about 37 % of a full run.
- **EXPECTED:** Same numbers, faster.
- **ACTUAL:** RK4 is bit-identical (max |Δ| = 0) and 1.9× faster. OE takes 7.8 s instead of 28.4 s, with
  estimates within 1e-6 (2e-5 σ). Sensitivities are bit-identical, 0.4 s instead of 6.2 s.
  **Full preset: every metric bit-identical to E00, runtime 150 s → 80 s.** Test suite 169 s → 57 s.
- **DECISION:** Keep (Agent 3: the equation of motion is still written in exactly one place).

---

## C3 / E03 — Honest parameter covariance (Agents 1 + 4; challenged by 3)

- **CHANGE:** `identifiability.parameter_covariance` is the sandwich covariance with per-channel ([V; x])
  noise variance. It raises if residual and S rows are misaligned. The pipeline builds S at the
  OE training samples (`oe_info["t"]`). Metrics `*_gn_sigma` are renamed to `*_sigma` (old values
  were wrong), and `kappa_sigma` is added (linear propagation).
- **WHY:** E00 combined S at 162 data times with residuals from 130 training points (σ 1.25× low) and
  pooled two channels whose normalized noise differs about 5×.
- **EXPECTED:** z = err/σ has unit spread across noise realizations.
- **ACTUAL (Monte-Carlo, 40 seeds, `scripts/oe_montecarlo.py`, `E03_covariance/oe_mc_unweighted.json`):**

  | σ variant | z-std μ / c_D / η | 68 % cover | 95 % cover |
  |---|---|---|---|
  | E00 code | 1.68 / 1.73 / 1.68 | 0.43–0.45 | 0.725 |
  | pooled, consistent | 1.35 / 1.40 / 1.35 | 0.50–0.60 | 0.78–0.80 |
  | **sandwich (C3)** | **1.02 / 1.08 / 1.02** | 0.65–0.68 | 0.90–0.95 |

  Empirical OE RMSE: μ 0.0249, c_D 0.0628, η 0.0662 (bias −0.005 / −0.001 / −0.015, all < 0.25 RMSE).
  Full preset (seed 7): estimates unchanged. σ = 0.0225 / 0.0534 / 0.0600, κ σ = 0.0084.
  z = −0.18 / 0.63 / −0.19 and κ z = 0.36. UQ σ(x_LOF) 0.497 → 0.513 m. Tests 44/44.
- **DECISION:** Keep. The E00 "all within 1σ" claim still holds, but with σ that are 1.7× larger than
  the ones printed in v1.
