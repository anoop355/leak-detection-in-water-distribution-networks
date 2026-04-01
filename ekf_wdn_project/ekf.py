from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from config import EstimatorConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class EKFStepResult:
    x_pred: np.ndarray
    p_pred: np.ndarray
    y_pred: np.ndarray
    residual: np.ndarray
    innovation_covariance: np.ndarray
    kalman_gain: np.ndarray
    x_upd: np.ndarray
    p_upd: np.ndarray
    normalized_residual: np.ndarray


class ExtendedKalmanFilter:
    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        config: EstimatorConfig,
    ) -> None:
        self.x = np.asarray(initial_state, dtype=float)
        self.P = np.asarray(initial_covariance, dtype=float)
        self.Q = np.asarray(process_noise, dtype=float)
        self.R = np.asarray(measurement_noise, dtype=float)
        self.config = config

    def step(
        self,
        measurement: np.ndarray,
        transition_function,
        measurement_function,
        transition_jacobian_function,
        measurement_jacobian_function,
    ) -> EKFStepResult:
        x_pred = np.asarray(transition_function(self.x), dtype=float)
        F = np.asarray(transition_jacobian_function(self.x), dtype=float)
        p_pred = F @ self.P @ F.T + self.Q
        p_pred = self._regularize_covariance(p_pred)

        y_pred = np.asarray(measurement_function(x_pred), dtype=float)
        residual = np.asarray(measurement, dtype=float) - y_pred
        H = np.asarray(measurement_jacobian_function(x_pred), dtype=float)
        innovation_covariance = H @ p_pred @ H.T + self.R
        innovation_covariance = self._regularize_covariance(
            innovation_covariance,
            epsilon=self.config.innovation_regularization,
        )

        if self.config.bad_data_sigma_limit is not None:
            residual = self._gate_residual(residual, innovation_covariance)

        kalman_gain = self._compute_kalman_gain(p_pred, H, innovation_covariance)
        x_upd = x_pred + kalman_gain @ residual
        x_upd = self._apply_state_constraints(x_upd)

        identity = np.eye(self.P.shape[0], dtype=float)
        p_upd = (identity - kalman_gain @ H) @ p_pred @ (identity - kalman_gain @ H).T
        p_upd = p_upd + kalman_gain @ self.R @ kalman_gain.T
        p_upd = self._regularize_covariance(p_upd)

        normalized_residual = residual / np.sqrt(np.maximum(np.diag(innovation_covariance), 1e-12))

        self.x = x_upd
        self.P = p_upd

        return EKFStepResult(
            x_pred=x_pred,
            p_pred=p_pred,
            y_pred=y_pred,
            residual=residual,
            innovation_covariance=innovation_covariance,
            kalman_gain=kalman_gain,
            x_upd=x_upd,
            p_upd=p_upd,
            normalized_residual=normalized_residual,
        )

    def _compute_kalman_gain(
        self,
        p_pred: np.ndarray,
        measurement_jacobian: np.ndarray,
        innovation_covariance: np.ndarray,
    ) -> np.ndarray:
        try:
            innovation_inverse = np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            LOGGER.warning("Innovation covariance was singular; using pseudo-inverse.")
            innovation_inverse = np.linalg.pinv(innovation_covariance)
        return p_pred @ measurement_jacobian.T @ innovation_inverse

    def _apply_state_constraints(self, state: np.ndarray) -> np.ndarray:
        constrained = np.asarray(state, dtype=float).copy()
        demand_slice = slice(len(self.config.demand_nodes), None)
        constrained[demand_slice] = np.maximum(
            constrained[demand_slice],
            self.config.minimum_demand,
        )
        if self.config.maximum_demand is not None:
            constrained[demand_slice] = np.minimum(
                constrained[demand_slice],
                self.config.maximum_demand,
            )
        return constrained

    def _gate_residual(self, residual: np.ndarray, innovation_covariance: np.ndarray) -> np.ndarray:
        sigma = np.sqrt(np.maximum(np.diag(innovation_covariance), 1e-12))
        gated = residual.copy()
        mask = np.abs(gated) > (self.config.bad_data_sigma_limit * sigma)
        if np.any(mask):
            LOGGER.warning("Bad-data gate applied to measurement residual(s): %s", np.where(mask)[0].tolist())
            gated[mask] = 0.0
        return gated

    def _regularize_covariance(self, covariance: np.ndarray, epsilon: float | None = None) -> np.ndarray:
        eps = self.config.covariance_regularization if epsilon is None else epsilon
        regularized = 0.5 * (covariance + covariance.T)
        regularized = regularized + np.eye(regularized.shape[0], dtype=float) * eps
        return regularized
