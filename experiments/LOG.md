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
  The "+" combination changes it by 0.36 m. Tests: 41 pass.
- **DECISION:** Keep. Thesis and audit text are corrected in the v2 thesis sources (see T1).
