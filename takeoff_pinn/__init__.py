"""Reproducible hybrid PINN for UAV takeoff ground-roll identification.

NUMERICAL CONTRACT: this package computes in float64. Importing it sets
torch's default dtype to float64. PINN residuals, L-BFGS line searches, and
finite-difference Jacobians (output-error stage) are all unreliable in
float32 -- e.g. the default scipy least_squares FD step perturbs a float32
RK4 trajectory below its quantization, producing an exactly-zero Jacobian.
(The legacy thesis script v3 hit the same class of problem.)
"""
import torch

torch.set_default_dtype(torch.float64)
