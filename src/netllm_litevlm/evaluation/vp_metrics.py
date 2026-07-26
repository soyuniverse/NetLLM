from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np


def _as_float_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.size == 0:
        raise ValueError("metric inputs must not be empty")
    if not np.isfinite(result).all():
        raise ValueError("metric inputs must contain only finite values")
    return result


def _validate_pair(prediction: Any, target: Any):
    prediction_array = _as_float_array(prediction)
    target_array = _as_float_array(target)
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            "prediction and target shapes must match: "
            f"{prediction_array.shape} != {target_array.shape}"
        )
    return prediction_array, target_array


def _absolute_error(
    prediction: Any,
    target: Any,
    rotation_aware: bool = False,
    period: float = 360.0,
) -> np.ndarray:
    prediction_array, target_array = _validate_pair(prediction, target)
    error = np.abs(prediction_array - target_array)
    if rotation_aware:
        if period <= 0:
            raise ValueError("period must be positive")
        error = np.mod(error, period)
        error = np.minimum(error, period - error)
    return error


def mae(
    prediction: Any,
    target: Any,
    rotation_aware: bool = False,
    period: float = 360.0,
) -> float:
    return float(
        np.mean(
            _absolute_error(
                prediction,
                target,
                rotation_aware=rotation_aware,
                period=period,
            )
        )
    )


def rmse(
    prediction: Any,
    target: Any,
    rotation_aware: bool = False,
    period: float = 360.0,
) -> float:
    error = _absolute_error(
        prediction,
        target,
        rotation_aware=rotation_aware,
        period=period,
    )
    return float(np.sqrt(np.mean(np.square(error))))


def upstream_rmse(
    prediction: Any,
    target: Any,
    rotation: bool = False,
) -> float:
    """Reproduce upstream ``compute_rmse``, including its ignored rotation flag."""

    del rotation
    return rmse(prediction, target, rotation_aware=False)


def corrected_rotation_aware_rmse(
    prediction: Any,
    target: Any,
    period: float = 360.0,
) -> float:
    return rmse(prediction, target, rotation_aware=True, period=period)


def mean_angular_error(
    prediction: Any,
    target: Any,
    period: float = 360.0,
) -> float:
    """Mean coordinate-wise circular angular error, in input angle units."""

    return mae(prediction, target, rotation_aware=True, period=period)


@dataclass(frozen=True)
class VPMetrics:
    mae: Optional[float]
    rmse: Optional[float]
    upstream_rmse: Optional[float]
    corrected_rotation_aware_rmse: Optional[float]
    mean_angular_error: Optional[float]
    test_loss: Optional[float]
    metric_valid: bool
    metric_invalid_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_vp_metrics(
    prediction: Optional[Any],
    target: Optional[Any],
    test_loss: Optional[float] = None,
    checkpoint_available: bool = False,
    period: float = 360.0,
) -> VPMetrics:
    if not checkpoint_available:
        return VPMetrics(
            mae=None,
            rmse=None,
            upstream_rmse=None,
            corrected_rotation_aware_rmse=None,
            mean_angular_error=None,
            test_loss=None,
            metric_valid=False,
            metric_invalid_reason="trained checkpoint unavailable",
        )
    if prediction is None or target is None:
        raise ValueError(
            "prediction and target are required when checkpoint_available=True"
        )
    if test_loss is not None and not np.isfinite(float(test_loss)):
        raise ValueError("test_loss must be finite")

    corrected_rmse = corrected_rotation_aware_rmse(
        prediction, target, period=period
    )
    return VPMetrics(
        mae=mae(prediction, target, rotation_aware=True, period=period),
        rmse=corrected_rmse,
        upstream_rmse=upstream_rmse(prediction, target, rotation=True),
        corrected_rotation_aware_rmse=corrected_rmse,
        mean_angular_error=mean_angular_error(
            prediction, target, period=period
        ),
        test_loss=None if test_loss is None else float(test_loss),
        metric_valid=True,
        metric_invalid_reason=None,
    )
