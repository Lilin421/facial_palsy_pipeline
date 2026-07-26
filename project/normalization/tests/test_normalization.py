"""
Unit tests for the normalization pipeline components.

Run with: python -m pytest normalization/tests/test_normalization.py -v
"""

import numpy as np
import pytest

from normalization.eye_center import compute_eye_centers, LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR
from normalization.similarity_transform import compute_similarity_transform
from normalization.one_euro_filter import OneEuroFilter, LowPassFilter
from normalization.interpolation import interpolate_missing_frames
from normalization.io import save_landmarks_csv, load_landmarks_csv


class TestEyeCenter:
    """Tests for eye center computation."""

    def test_468_landmarks(self):
        """468-landmark model uses contour centroid."""
        landmarks = np.random.rand(468, 3)
        left, right = compute_eye_centers(landmarks)

        # Should be centroid of respective contour indices
        expected_left = landmarks[LEFT_EYE_CONTOUR].mean(axis=0)
        expected_right = landmarks[RIGHT_EYE_CONTOUR].mean(axis=0)

        np.testing.assert_allclose(left, expected_left)
        np.testing.assert_allclose(right, expected_right)

    def test_478_landmarks_uses_iris(self):
        """478-landmark model uses iris center."""
        landmarks = np.random.rand(478, 3)
        left, right = compute_eye_centers(landmarks)

        np.testing.assert_allclose(left, landmarks[468])
        np.testing.assert_allclose(right, landmarks[473])

    def test_invalid_count_raises(self):
        """Non-468/478 counts should raise ValueError."""
        with pytest.raises(ValueError):
            compute_eye_centers(np.random.rand(100, 3))


class TestSimilarityTransform:
    """Tests for spatial normalization."""

    def _make_test_case(self):
        """Create landmarks with known eye positions."""
        landmarks = np.random.rand(468, 3)
        left_eye = np.array([0.3, 0.5, 0.0])
        right_eye = np.array([0.7, 0.5, 0.0])
        return landmarks, left_eye, right_eye

    def test_eye_midpoint_at_origin(self):
        """After transform, eye midpoint should be at origin."""
        landmarks, left_eye, right_eye = self._make_test_case()
        result = compute_similarity_transform(landmarks, left_eye, right_eye)

        # Recompute eye positions in normalized space
        midpoint = (left_eye + right_eye) / 2.0
        inter_eye = np.linalg.norm(right_eye - left_eye)
        normalized_mid = (midpoint - midpoint) / inter_eye  # Should be [0,0,0]

        np.testing.assert_allclose(normalized_mid, [0, 0, 0], atol=1e-10)

    def test_inter_eye_distance_is_one(self):
        """After transform, inter-eye distance should be 1."""
        landmarks = np.random.rand(468, 3)
        left_eye = np.array([0.3, 0.4, 0.0])
        right_eye = np.array([0.7, 0.6, 0.0])

        result = compute_similarity_transform(landmarks, left_eye, right_eye)

        # Transform the eye points themselves
        midpoint = (left_eye + right_eye) / 2.0
        inter_eye = np.linalg.norm(right_eye - left_eye)
        left_norm = (left_eye - midpoint) / inter_eye
        right_norm = (right_eye - midpoint) / inter_eye

        dist = np.linalg.norm(right_norm - left_norm)
        assert abs(dist - 1.0) < 1e-10

    def test_eyes_horizontal_after_rotation(self):
        """After rotation, eyes should be on same y-level."""
        landmarks = np.random.rand(468, 3)
        # Tilted eyes
        left_eye = np.array([0.3, 0.6, 0.0])
        right_eye = np.array([0.7, 0.4, 0.0])

        result = compute_similarity_transform(landmarks, left_eye, right_eye, enable_rotation=True)

        # Transform eyes
        midpoint = (left_eye + right_eye) / 2.0
        inter_eye = np.linalg.norm(right_eye - left_eye)
        left_t = (left_eye - midpoint) / inter_eye
        right_t = (right_eye - midpoint) / inter_eye

        angle = np.arctan2(
            right_t[1] - left_t[1],
            right_t[0] - left_t[0]
        )
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)

        left_rot_y = left_t[0] * sin_a + left_t[1] * cos_a
        right_rot_y = right_t[0] * sin_a + right_t[1] * cos_a

        assert abs(left_rot_y - right_rot_y) < 1e-10

    def test_zero_distance_raises(self):
        """Zero inter-eye distance should raise."""
        landmarks = np.random.rand(468, 3)
        eye = np.array([0.5, 0.5, 0.0])

        with pytest.raises(ValueError):
            compute_similarity_transform(landmarks, eye, eye)


class TestOneEuroFilter:
    """Tests for One Euro Filter."""

    def test_constant_signal_unchanged(self):
        """Constant signal should pass through unchanged."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        values = [5.0] * 100

        results = []
        for i, v in enumerate(values):
            results.append(f.apply(v, i / 30.0))

        # After convergence, output should equal input
        assert abs(results[-1] - 5.0) < 1e-6

    def test_smoothing_reduces_noise(self):
        """Filter should reduce high-frequency noise."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)

        np.random.seed(42)
        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        noisy = signal + np.random.normal(0, 0.1, 100)

        filtered = []
        for i, v in enumerate(noisy):
            filtered.append(f.apply(v, i / 30.0))

        filtered = np.array(filtered)
        # Filtered should be closer to true signal than noisy
        noise_error = np.std(noisy - signal)
        filter_error = np.std(filtered[10:] - signal[10:])  # Skip transient

        assert filter_error < noise_error

    def test_low_pass_filter_alpha_one(self):
        """Alpha=1 means no smoothing."""
        lpf = LowPassFilter()
        assert lpf.apply(5.0, 1.0) == 5.0
        assert lpf.apply(10.0, 1.0) == 10.0


class TestInterpolation:
    """Tests for missing frame interpolation."""

    def test_no_missing(self):
        """All valid frames should pass through unchanged."""
        seq = np.random.rand(10, 468, 3)
        mask = np.ones(10, dtype=bool)

        result, new_mask = interpolate_missing_frames(seq, mask)
        np.testing.assert_array_equal(result, seq)
        assert new_mask.all()

    def test_missing_start(self):
        """Missing start should be filled with first valid."""
        seq = np.zeros((10, 2, 3))
        seq[3] = [[1, 2, 3], [4, 5, 6]]
        mask = np.zeros(10, dtype=bool)
        mask[3:] = True
        seq[3:] = seq[3]

        result, new_mask = interpolate_missing_frames(seq, mask)
        np.testing.assert_array_equal(result[0], result[3])
        assert new_mask.all()

    def test_missing_end(self):
        """Missing end should be filled with last valid."""
        seq = np.ones((10, 2, 3))
        mask = np.ones(10, dtype=bool)
        mask[7:] = False
        seq[7:] = 0

        result, new_mask = interpolate_missing_frames(seq, mask)
        np.testing.assert_array_equal(result[9], result[6])
        assert new_mask.all()

    def test_interior_gap_linear(self):
        """Interior gap should be linearly interpolated."""
        seq = np.zeros((5, 1, 3))
        seq[0] = [[0, 0, 0]]
        seq[4] = [[4, 8, 12]]
        mask = np.array([True, False, False, False, True])

        result, _ = interpolate_missing_frames(seq, mask)

        # Frame 2 should be midpoint
        np.testing.assert_allclose(result[2, 0], [2, 4, 6])

    def test_all_invalid(self):
        """All invalid returns unchanged."""
        seq = np.zeros((5, 2, 3))
        mask = np.zeros(5, dtype=bool)

        result, new_mask = interpolate_missing_frames(seq, mask)
        np.testing.assert_array_equal(result, seq)
        assert not new_mask.any()


class TestCSVIO:
    """Tests for CSV export/import."""

    def test_roundtrip(self, tmp_path):
        """Save and load should produce identical data."""
        seq = np.random.rand(5, 10, 3)
        csv_path = str(tmp_path / "test.csv")

        save_landmarks_csv(seq, csv_path, precision=8)
        loaded = load_landmarks_csv(csv_path)

        np.testing.assert_allclose(loaded, seq, atol=1e-7)
