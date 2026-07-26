"""
Visualization utilities for landmark normalization results.

Generates comparison plots (before/after normalization) and
trajectory plots (before/after temporal smoothing).
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Representative landmarks for trajectory visualization
# nose tip, left eye corner, right eye corner, chin, left mouth corner
REPRESENTATIVE_LANDMARKS = [1, 33, 263, 152, 61]


def plot_frame_comparison(
    raw_landmarks: NDArray[np.float64],
    normalized_landmarks: NDArray[np.float64],
    frame_idx: int,
    output_path: str,
) -> None:
    """Plot before/after normalization for a single frame.

    Args:
        raw_landmarks: Shape (N, 3) — original coordinates.
        normalized_landmarks: Shape (N, 3) — normalized coordinates.
        frame_idx: Frame index (for title).
        output_path: Path to save the figure.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping visualization.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Before
    axes[0].scatter(raw_landmarks[:, 0], -raw_landmarks[:, 1], s=1, c="blue")
    axes[0].set_title(f"Frame {frame_idx} — Before Normalization")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    # After
    axes[1].scatter(normalized_landmarks[:, 0], -normalized_landmarks[:, 1], s=1, c="green")
    axes[1].set_title(f"Frame {frame_idx} — After Normalization")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trajectory_comparison(
    raw_sequence: NDArray[np.float64],
    smoothed_sequence: NDArray[np.float64],
    output_path: str,
    landmark_indices: list[int] | None = None,
) -> None:
    """Plot x/y trajectories before and after smoothing for representative landmarks.

    Args:
        raw_sequence: Shape (T, N, 3) before smoothing.
        smoothed_sequence: Shape (T, N, 3) after smoothing.
        output_path: Path to save the figure.
        landmark_indices: Which landmarks to plot. Defaults to representative set.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping visualization.")
        return

    if landmark_indices is None:
        landmark_indices = REPRESENTATIVE_LANDMARKS

    T = raw_sequence.shape[0]
    frames = np.arange(T)
    n_landmarks = len(landmark_indices)

    fig, axes = plt.subplots(n_landmarks, 2, figsize=(14, 3 * n_landmarks))
    if n_landmarks == 1:
        axes = axes[np.newaxis, :]

    for i, lm_idx in enumerate(landmark_indices):
        # X coordinate
        axes[i, 0].plot(frames, raw_sequence[:, lm_idx, 0], alpha=0.5, label="raw", linewidth=0.8)
        axes[i, 0].plot(frames, smoothed_sequence[:, lm_idx, 0], label="smoothed", linewidth=1.0)
        axes[i, 0].set_ylabel(f"LM {lm_idx} — x")
        axes[i, 0].legend(fontsize=7)

        # Y coordinate
        axes[i, 1].plot(frames, raw_sequence[:, lm_idx, 1], alpha=0.5, label="raw", linewidth=0.8)
        axes[i, 1].plot(frames, smoothed_sequence[:, lm_idx, 1], label="smoothed", linewidth=1.0)
        axes[i, 1].set_ylabel(f"LM {lm_idx} — y")
        axes[i, 1].legend(fontsize=7)

    axes[-1, 0].set_xlabel("Frame")
    axes[-1, 1].set_xlabel("Frame")
    fig.suptitle("Trajectory: Before vs After Smoothing")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
