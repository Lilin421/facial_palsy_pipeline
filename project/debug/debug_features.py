"""
Debug script — extracts features from landmarks and generates visualizations.

Usage:
    python debug/debug_features.py                    # All features
    python debug/debug_features.py --feature EAR_L    # Single feature plot
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from feature_extraction import extract_all_features


def plot_all_features(df, output_path: str) -> None:
    """Plot all features as subplots in a single figure.

    Args:
        df: Features DataFrame.
        output_path: Path to save the figure.
    """
    feature_cols = [c for c in df.columns if c != "frame"]
    n_features = len(feature_cols)

    fig, axes = plt.subplots(n_features, 1, figsize=(14, 2.5 * n_features), sharex=True)
    if n_features == 1:
        axes = [axes]

    frames = df["frame"].values

    for i, col in enumerate(feature_cols):
        axes[i].plot(frames, df[col].values, linewidth=0.8)
        axes[i].set_ylabel(col, fontsize=8)
        axes[i].tick_params(labelsize=7)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame")
    fig.suptitle("Feature Overview", fontsize=12)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_single_feature(df, feature_name: str, output_path: str) -> None:
    """Plot a single feature time series.

    Args:
        df: Features DataFrame.
        feature_name: Column name to plot.
        output_path: Path to save the figure.
    """
    if feature_name not in df.columns:
        available = [c for c in df.columns if c != "frame"]
        print(f"ERROR: '{feature_name}' not found. Available: {available}")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["frame"].values, df[feature_name].values, linewidth=1.0)
    ax.set_xlabel("Frame")
    ax.set_ylabel(feature_name)
    ax.set_title(feature_name)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature extraction debug")
    parser.add_argument("--feature", type=str, default=None,
                        help="Plot a single feature (e.g., EAR_L)")
    parser.add_argument("--input", type=str, default="output/normalization/landmarks.csv",
                        help="Path to landmarks CSV")
    args = parser.parse_args()

    output_dir = "debug"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract features
    print("Loading landmarks and extracting features...")
    df = extract_all_features(csv_path=args.input)

    # Save features CSV
    csv_out = f"{output_dir}/features.csv"
    df.to_csv(csv_out, index=False)
    print(f"Features saved: {csv_out} ({len(df)} frames, {len(df.columns)-1} features)")

    # Plot
    if args.feature:
        plot_single_feature(df, args.feature, f"{output_dir}/{args.feature}.png")
    else:
        plot_all_features(df, f"{output_dir}/features_overview.png")


if __name__ == "__main__":
    main()
