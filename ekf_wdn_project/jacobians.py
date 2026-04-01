from __future__ import annotations

import numpy as np

from config import EstimatorConfig


def numerical_jacobian(func, x: np.ndarray, config: EstimatorConfig) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    fx = np.asarray(func(x), dtype=float)
    jac = np.zeros((fx.size, x.size), dtype=float)

    for idx in range(x.size):
        step = max(abs(x[idx]) * config.jacobian_relative_step, config.jacobian_min_step)
        delta = np.zeros_like(x)
        delta[idx] = step

        f_plus = np.asarray(func(x + delta), dtype=float)
        f_minus = np.asarray(func(x - delta), dtype=float)
        jac[:, idx] = (f_plus - f_minus) / (2.0 * step)

    return jac
