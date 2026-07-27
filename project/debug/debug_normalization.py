"""
Debug script — runs the full landmark normalization pipeline.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from normalization import NormalizationConfig, normalize_video_landmarks


def main() -> None:
    """Run normalization pipeline for debugging."""
    output_dir = "output/normalization"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/debug.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Create detector
    base_options = python.BaseOptions(
        model_asset_path="C:/Users/lilia/Desktop/agentic_ai/test/pof_ver2/project/landmark/face_landmarker.task"
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Config
    config = NormalizationConfig(
        enable_rotation=True,
        enable_temporal_smoothing=True,
        enable_interpolation=True,
        one_euro_min_cutoff=1.0,
        one_euro_beta=0.0,
        one_euro_derivate_cutoff=1.0,
    )

    # Run
    normalized = normalize_video_landmarks(
        video_path="C:/Users/lilia/Desktop/agentic_ai/test/pof_ver2/project/data/test_2.mp4",
        detector=detector,
        output_csv=f"{output_dir}/landmarks.csv",
        config=config,
    )

    detector.close()

    T, N, _ = normalized.shape
    print(f"Done: {T} frames, {N} landmarks per frame.")
    print(f"CSV saved to: {output_dir}/landmarks.csv")


if __name__ == "__main__":
    main()
