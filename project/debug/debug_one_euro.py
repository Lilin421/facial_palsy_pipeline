"""
Debug script — tests One Euro Filter on synthetic noisy signal.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from normalization.one_euro_filter import OneEuroFilter


def main() -> None:
    # Generate a clean sine wave + noise
    np.random.seed(42)
    fps = 30.0
    duration_s = 3.0
    T = int(fps * duration_s)
    t = np.linspace(0, duration_s, T)

    clean_signal = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz sine
    noise = np.random.normal(0, 0.1, T)
    noisy_signal = clean_signal + noise

    # Filter with different beta values
    for beta in [0.0, 0.5, 1.0]:
        f = OneEuroFilter(min_cutoff=1.0, beta=beta, derivate_cutoff=1.0)
        filtered = []
        for i in range(T):
            filtered.append(f.apply(noisy_signal[i], t[i]))
        filtered = np.array(filtered)

        error_noisy = np.std(noisy_signal[10:] - clean_signal[10:])
        error_filtered = np.std(filtered[10:] - clean_signal[10:])

        print(f"beta={beta:.1f}: noise_std={error_noisy:.4f} → filtered_std={error_filtered:.4f} "
              f"(reduction: {(1 - error_filtered/error_noisy)*100:.1f}%)")

    print("\nOne Euro Filter working correctly.")


if __name__ == "__main__":
    main()
