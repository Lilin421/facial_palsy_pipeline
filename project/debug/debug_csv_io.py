"""
Debug script — tests CSV save/load roundtrip.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from normalization.io import save_landmarks_csv, load_landmarks_csv


def main() -> None:
    output_dir = Path("output/normalization")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = str(output_dir / "test_roundtrip.csv")

    # Create synthetic data
    T, N = 5, 10
    original = np.random.rand(T, N, 3)

    # Save
    save_landmarks_csv(original, csv_path, precision=8)
    print(f"Saved: {csv_path}")

    # Load
    loaded = load_landmarks_csv(csv_path)
    print(f"Loaded shape: {loaded.shape}")

    # Compare
    max_diff = np.max(np.abs(loaded - original))
    print(f"Max roundtrip error: {max_diff:.2e}")

    if max_diff < 1e-7:
        print("PASS: roundtrip within tolerance.")
    else:
        print("FAIL: roundtrip error too large.")


if __name__ == "__main__":
    main()
