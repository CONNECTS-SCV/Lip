"""Common math utilities - sigmoid, normalization, transforms."""

from __future__ import annotations

import math


def sigmoid(x: float, center: float, steepness: float = 1.0) -> float:
    """Sigmoid normalization: maps x to (0, 1), centered at `center`.

    Returns high values when x < center, low values when x > center.
    """
    z = steepness * (x - center)
    # Clamp to avoid overflow
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(z))


def normalize_score(
    value: float, low: float, high: float, *, clip: bool = True,
) -> float:
    """Linear normalization: maps [low, high] -> [0, 1].

    If clip is True, values outside [low, high] are clamped to [0, 1].
    """
    if abs(high - low) < 1e-12:
        return 0.5
    score = (value - low) / (high - low)
    if clip:
        score = max(0.0, min(1.0, score))
    return score


# ---------------------------------------------------------------------------
# REINVENT4 transform dict builders (for TOML config generation)
# ---------------------------------------------------------------------------

def reinvent_double_sigmoid(low: float, high: float,
                            coef_div: float = 100.0,
                            coef_si: float = 150.0,
                            coef_se: float = 150.0) -> dict:
    """Build a REINVENT4 double_sigmoid transform dict."""
    return {
        "type": "double_sigmoid",
        "low": low,
        "high": high,
        "coef_div": coef_div,
        "coef_si": coef_si,
        "coef_se": coef_se,
    }


def reinvent_reverse_sigmoid(high: float, low: float = 0.0,
                              k: float = 0.4) -> dict:
    """Build a REINVENT4 reverse_sigmoid transform dict."""
    return {"type": "reverse_sigmoid", "high": high, "low": low, "k": k}


def reinvent_sigmoid(low: float, high: float = 1.0,
                     k: float = 0.4) -> dict:
    """Build a REINVENT4 sigmoid transform dict."""
    return {"type": "sigmoid", "low": low, "high": high, "k": k}
