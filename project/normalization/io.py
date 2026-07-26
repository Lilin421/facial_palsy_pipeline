"""
I/O utilities for landmark data — CSV export and import.

CSV format:
    frame_id, landmark_id, x, y, z

All coordinates are the final normalized values after the full pipeline.
"""

import csv
import numpy as np
from numpy.typing import NDArray
from pathlib import Path


def save_landmarks_csv(
    sequence: NDArray[np.float64],
    output_path: str,
    precision: int = 6,
) -> None:
    """Save normalized landmark sequence to CSV.

    Args:
        sequence: Shape (T, N, 3) — T frames, N landmarks, xyz.
        output_path: Path to output CSV file.
        precision: Number of decimal places.

    Output format:
        frame_id,landmark_id,x,y,z
        0,0,0.123456,-0.234567,0.001234
        0,1,0.124000,-0.230000,0.001300
        ...
    """
    T, N, _ = sequence.shape
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fmt = f"{{:.{precision}f}}"

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "landmark_id", "x", "y", "z"])

        for t in range(T):
            for n in range(N):
                x, y, z = sequence[t, n]
                writer.writerow([
                    t,
                    n,
                    fmt.format(x),
                    fmt.format(y),
                    fmt.format(z),
                ])


def load_landmarks_csv(csv_path: str) -> NDArray[np.float64]:
    """Load landmark CSV back into numpy array.

    Args:
        csv_path: Path to CSV file.

    Returns:
        Shape (T, N, 3) array.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If CSV format is invalid.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data = np.loadtxt(path, delimiter=",", skiprows=1)

    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError(f"Expected 5 columns, got shape {data.shape}")

    frame_ids = data[:, 0].astype(int)
    landmark_ids = data[:, 1].astype(int)

    T = frame_ids.max() + 1
    N = landmark_ids.max() + 1

    sequence = np.zeros((T, N, 3), dtype=np.float64)
    sequence[frame_ids, landmark_ids] = data[:, 2:5]

    return sequence
