"""
Feature extraction pipeline — orchestrates all extractors.

Loads normalized landmarks, computes the reference distance,
and calls each feature extractor to produce a combined DataFrame.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numpy.typing import NDArray

from .geometry import interocular_distance
from .eyebrow import extract_brow_features
from .eye import extract_eye_features
from .mouth import extract_mouth_features
from .cheek import extract_cheek_features
from .jaw import extract_jaw_features


def load_landmarks(csv_path: str) -> NDArray[np.float64]:
    """Load normalized landmark CSV into a (T, N, 3) array.

    Expected CSV format: frame_id, landmark_id, x, y, z

    Args:
        csv_path: Path to landmarks CSV.

    Returns:
        Shape (T, N, 3) numpy array.

    Raises:
        FileNotFoundError: If CSV doesn't exist.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Landmarks CSV not found: {csv_path}")

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    frame_ids = data[:, 0].astype(int)
    landmark_ids = data[:, 1].astype(int)

    T = frame_ids.max() + 1
    N = landmark_ids.max() + 1

    landmarks = np.zeros((T, N, 3), dtype=np.float64)
    landmarks[frame_ids, landmark_ids] = data[:, 2:5]

    return landmarks


def extract_all_features(
    landmarks: NDArray[np.float64] | None = None,
    csv_path: str | None = None,
) -> pd.DataFrame:
    """Extract all facial features from normalized landmarks.

    Provide either landmarks array directly or csv_path to load from file.

    Args:
        landmarks: Shape (T, N, 3) array. If None, loads from csv_path.
        csv_path: Path to landmarks CSV. Used if landmarks is None.

    Returns:
        DataFrame with frame index and all feature columns.
    """
    if landmarks is None:
        if csv_path is None:
            raise ValueError("Provide either landmarks array or csv_path.")
        landmarks = load_landmarks(csv_path)

    T = landmarks.shape[0]

    # Compute reference distance for normalization
    reference = interocular_distance(landmarks)

    # Extract feature groups
    brow_df = extract_brow_features(landmarks, reference)
    eye_df = extract_eye_features(landmarks, reference)
    mouth_df = extract_mouth_features(landmarks, reference)
    cheek_df = extract_cheek_features(landmarks, reference)
    jaw_df = extract_jaw_features(landmarks, reference)

    # Combine all
    features = pd.DataFrame({"frame": np.arange(T)})
    features = pd.concat(
        [features, brow_df, eye_df, mouth_df, cheek_df, jaw_df],
        axis=1,
    )

    return features
