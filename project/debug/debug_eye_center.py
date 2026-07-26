"""
Debug script — tests eye center computation on a single video frame.
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
        print("ERROR: Cannot read video frame.")
        detector.close()
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, 0)
    detector.close()

    if not result.face_landmarks:
        print("No face detected in frame 0.")
        return

    landmarks = mediapipe_to_numpy(result.face_landmarks[0])
    left_eye, right_eye = compute_eye_centers(landmarks)

    print(f"Landmark count: {landmarks.shape[0]}")
    print(f"Left eye center:  {left_eye}")
    print(f"Right eye center: {right_eye}")
    print(f"Midpoint:         {(left_eye + right_eye) / 2}")
    print(f"Inter-eye dist:   {np.linalg.norm(right_eye - left_eye):.6f}")


if __name__ == "__main__":
    main()
