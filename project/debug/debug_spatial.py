"""
Debug script — tests spatial normalization (translation + scale + rotation)
on a single frame and prints sanity check results.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from normalization.eye_center import compute_eye_centers
from normalization.similarity_transform import compute_similarity_transform
from normalization.utils import mediapipe_to_numpy


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load detector
    base_options = python.BaseOptions(model_asset_path="landmark/face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Read first frame
    cap = cv2.VideoCapture("data/with_audio.mp4")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Cannot read video.")
        detector.close()
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, 0)
    detector.close()

    if not result.face_landmarks:
        print("No face detected.")
        return

    landmarks = mediapipe_to_numpy(result.face_landmarks[0])
    left_eye, right_eye = compute_eye_centers(landmarks)

    # Apply spatial normalization
    normalized = compute_similarity_transform(landmarks, left_eye, right_eye, enable_rotation=True)

    # Sanity checks on normalized result
    norm_left, norm_right = compute_eye_centers(normalized)
    midpoint = (norm_left + norm_right) / 2.0
    inter_eye = np.linalg.norm(norm_right - norm_left)
    eye_y_diff = abs(norm_left[1] - norm_right[1])

    print("=== Spatial Normalization Sanity Checks ===")
    print(f"Eye midpoint (should be ~0):   {midpoint}")
    print(f"Inter-eye distance (should be ~1): {inter_eye:.8f}")
    print(f"Eye y-difference (should be ~0):   {eye_y_diff:.8f}")
    print(f"Any NaN: {np.any(np.isnan(normalized))}")
    print(f"Landmark shape: {normalized.shape}")


if __name__ == "__main__":
    main()
