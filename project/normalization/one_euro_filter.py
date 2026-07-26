"""
One Euro Filter — Python implementation following MediaPipe's official source.

References:
    - Original paper: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
      for Noisy Input in Interactive Systems", CHI 2012.
    - MediaPipe source:
      mediapipe/util/filtering/one_euro_filter.cc
      mediapipe/util/filtering/low_pass_filter.cc
      mediapipe/calculators/util/landmarks_smoothing_calculator.cc

MediaPipe implementation details:
    - Uses a first-order low-pass filter (exponential smoothing)
    - Alpha is computed adaptively based on signal speed
    - The derivative is also smoothed with a separate low-pass filter
    - When frequency is unknown, it is estimated from timestamps

    The key formula:
        alpha = 1 / (1 + tau / Te)
    where:
        tau = 1 / (2 * pi * fc)
        Te = 1 / frequency
        fc = min_cutoff + beta * |derivative|

    This means:
        - At low speed: fc ≈ min_cutoff → heavy smoothing
        - At high speed: fc increases → less lag
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field


def _smoothing_factor(te: float, cutoff: float) -> float:
    """Compute alpha (smoothing factor) from time period and cutoff frequency.

    This follows MediaPipe's low_pass_filter.cc implementation.

    Args:
        te: Time period between samples (1/frequency).
        cutoff: Cutoff frequency in Hz.

    Returns:
        Alpha value in [0, 1].
    """
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / te)


class LowPassFilter:
    """First-order IIR low-pass filter (exponential smoothing).

    Replicates mediapipe/util/filtering/low_pass_filter.cc
    """

    def __init__(self) -> None:
        self._initialized: bool = False
        self._raw_value: float = 0.0
        self._stored_value: float = 0.0

    @property
    def last_value(self) -> float:
        return self._stored_value

    def apply(self, value: float, alpha: float) -> float:
        """Apply low-pass filter with given alpha.

        Args:
            value: New raw input value.
            alpha: Smoothing factor in [0, 1]. Higher = less smoothing.

        Returns:
            Filtered value.
        """
        if not self._initialized:
            self._stored_value = value
            self._initialized = True
        else:
            self._stored_value = alpha * value + (1.0 - alpha) * self._stored_value
        self._raw_value = value
        return self._stored_value

    def has_last_raw_value(self) -> bool:
        return self._initialized

    @property
    def last_raw_value(self) -> float:
        return self._raw_value

    def reset(self) -> None:
        self._initialized = False


class OneEuroFilter:
    """One Euro Filter for a single scalar signal.

    Replicates mediapipe/util/filtering/one_euro_filter.cc
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        derivate_cutoff: float = 1.0,
    ) -> None:
        """Initialize One Euro Filter.

        Args:
            min_cutoff: Minimum cutoff frequency (Hz). Controls smoothing at low speed.
            beta: Speed coefficient. Controls lag reduction at high speed.
            derivate_cutoff: Cutoff for the derivative low-pass filter (Hz).
        """
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._derivate_cutoff = derivate_cutoff
        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._frequency: float = 0.0
        self._last_timestamp: float = -1.0

    def apply(self, value: float, timestamp_s: float) -> float:
        """Filter a single value at the given timestamp.

        Args:
            value: Raw input value.
            timestamp_s: Current timestamp in seconds.

        Returns:
            Filtered value.
        """
        # Estimate frequency from timestamps
        if self._last_timestamp >= 0.0:
            dt = timestamp_s - self._last_timestamp
            if dt > 1e-9:
                self._frequency = 1.0 / dt
        self._last_timestamp = timestamp_s

        # On first call, frequency may be 0; use a safe default
        if self._frequency < 1e-9:
            self._frequency = 30.0  # Assume ~30fps as fallback

        te = 1.0 / self._frequency

        # Compute derivative (speed)
        if self._x_filter.has_last_raw_value():
            dx = (value - self._x_filter.last_raw_value) / te
        else:
            dx = 0.0

        # Smooth the derivative
        d_alpha = _smoothing_factor(te, self._derivate_cutoff)
        dx_smooth = self._dx_filter.apply(dx, d_alpha)

        # Adaptive cutoff based on speed
        cutoff = self._min_cutoff + self._beta * abs(dx_smooth)

        # Smooth the value
        alpha = _smoothing_factor(te, cutoff)
        return self._x_filter.apply(value, alpha)

    def reset(self) -> None:
        """Reset filter state."""
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_timestamp = -1.0
        self._frequency = 0.0
