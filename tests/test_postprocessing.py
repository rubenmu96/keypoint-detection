"""
Edge-case tests for the keypoint post-processing pipeline.

Covers:
  - filter_low_probabilities  – confidence thresholding
  - overlapping_kps           – proximity-based deduplication
  - num_kps_req               – minimum-count / required-index guard
  - kps_postprocessor         – full pipeline integration
"""
import numpy as np
import pytest

from src.inference.processing import (
    filter_low_probabilities,
    overlapping_kps,
    num_kps_req,
    kps_postprocessor,
)

# Shorthand: a keypoint that is explicitly invalid
INVALID = np.array([-1.0, -1.0])


def _reshape(arr):
    """Flatten then reshape to (N, 2) for easy indexing in assertions."""
    return arr.reshape(-1, 2)


# ---------------------------------------------------------------------------
# filter_low_probabilities
# ---------------------------------------------------------------------------

class TestFilterLowProbabilities:
    def test_below_threshold_becomes_invalid(self):
        kps = np.array([[100.0, 200.0], [300.0, 400.0]])
        result = _reshape(filter_low_probabilities(kps, [0.3, 0.8], threshold=0.5))
        np.testing.assert_array_equal(result[0], INVALID)
        np.testing.assert_array_equal(result[1], [300.0, 400.0])

    def test_above_threshold_unchanged(self):
        kps = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = _reshape(filter_low_probabilities(kps, [0.9, 0.95], threshold=0.5))
        np.testing.assert_array_equal(result[0], [10.0, 20.0])
        np.testing.assert_array_equal(result[1], [30.0, 40.0])

    def test_all_below_threshold(self):
        kps = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = _reshape(filter_low_probabilities(kps, [0.1, 0.2], threshold=0.5))
        assert np.all(result == -1)

    def test_exactly_at_threshold_is_kept(self):
        """Value equal to threshold is not filtered out (uses < not <=)."""
        kps = np.array([[50.0, 60.0]])
        result = _reshape(filter_low_probabilities(kps, [0.5], threshold=0.5))
        np.testing.assert_array_equal(result[0], [50.0, 60.0])

    def test_flat_input_format(self):
        """Also accepts flat [x1, y1, x2, y2, ...] format and returns same."""
        kps = np.array([100.0, 200.0, 300.0, 400.0])
        result = filter_low_probabilities(kps, [0.1, 0.9], threshold=0.5)
        assert result.shape == (4,)
        assert result[0] == -1.0   # x1 filtered
        assert result[1] == -1.0   # y1 filtered
        assert result[2] == 300.0  # x2 kept
        assert result[3] == 400.0  # y2 kept

    def test_output_is_copy_not_in_place(self):
        """Original array must not be mutated."""
        kps = np.array([[100.0, 200.0]])
        original = kps.copy()
        filter_low_probabilities(kps, [0.1], threshold=0.5)
        np.testing.assert_array_equal(kps, original)


# ---------------------------------------------------------------------------
# overlapping_kps
# ---------------------------------------------------------------------------

class TestOverlappingKps:
    def test_lower_confidence_of_pair_is_removed(self):
        """When two keypoints are within pixel_distance, the weaker one goes."""
        kps = np.array([[100.0, 100.0], [101.0, 100.0]])   # distance = 1 < 10
        result = _reshape(overlapping_kps(kps, [0.9, 0.5], pixel_distance=10))
        np.testing.assert_array_equal(result[0], [100.0, 100.0])  # stronger kept
        np.testing.assert_array_equal(result[1], INVALID)          # weaker removed

    def test_higher_confidence_wins_when_reversed(self):
        """Same pair, confidences swapped – the other one gets removed."""
        kps = np.array([[100.0, 100.0], [101.0, 100.0]])
        result = _reshape(overlapping_kps(kps, [0.4, 0.9], pixel_distance=10))
        np.testing.assert_array_equal(result[0], INVALID)
        np.testing.assert_array_equal(result[1], [101.0, 100.0])

    def test_far_apart_keypoints_both_kept(self):
        """Keypoints beyond pixel_distance are both preserved."""
        kps = np.array([[0.0, 0.0], [500.0, 500.0]])
        result = _reshape(overlapping_kps(kps, [0.9, 0.8], pixel_distance=10))
        np.testing.assert_array_equal(result[0], [0.0, 0.0])
        np.testing.assert_array_equal(result[1], [500.0, 500.0])

    def test_exactly_at_distance_threshold_both_kept(self):
        """Distance == pixel_distance is NOT considered overlapping (strict <)."""
        kps = np.array([[0.0, 0.0], [10.0, 0.0]])   # distance = 10 exactly
        result = _reshape(overlapping_kps(kps, [0.9, 0.8], pixel_distance=10))
        np.testing.assert_array_equal(result[0], [0.0, 0.0])
        np.testing.assert_array_equal(result[1], [10.0, 0.0])

    def test_existing_invalid_keypoints_ignored(self):
        """Pre-existing (-1,-1) entries are not candidates for overlap removal."""
        kps = np.array([[-1.0, -1.0], [100.0, 100.0], [101.0, 100.0]])
        result = _reshape(overlapping_kps(kps, [0.99, 0.9, 0.5], pixel_distance=10))
        np.testing.assert_array_equal(result[0], INVALID)          # still invalid
        np.testing.assert_array_equal(result[1], [100.0, 100.0])   # kept
        np.testing.assert_array_equal(result[2], INVALID)           # removed due to overlap

    def test_single_keypoint_unchanged(self):
        """One keypoint cannot overlap with itself."""
        kps = np.array([[50.0, 50.0]])
        result = _reshape(overlapping_kps(kps, [0.9], pixel_distance=10))
        np.testing.assert_array_equal(result[0], [50.0, 50.0])

    def test_all_invalid_returns_all_invalid(self):
        """All-invalid input should pass through unchanged."""
        kps = np.array([[-1.0, -1.0], [-1.0, -1.0]])
        result = _reshape(overlapping_kps(kps, [0.9, 0.8], pixel_distance=10))
        assert np.all(result == -1)

    def test_three_keypoints_chain_overlap(self):
        """Three mutually close keypoints: only the highest-confidence survives."""
        # All within pixel_distance=10 of each other
        kps = np.array([[0.0, 0.0], [5.0, 0.0], [9.0, 0.0]])
        result = _reshape(overlapping_kps(kps, [0.5, 0.9, 0.3], pixel_distance=10))
        # Middle one (0.9) wins; others should be removed
        np.testing.assert_array_equal(result[1], [5.0, 0.0])
        np.testing.assert_array_equal(result[0], INVALID)
        np.testing.assert_array_equal(result[2], INVALID)


# ---------------------------------------------------------------------------
# num_kps_req
# ---------------------------------------------------------------------------

class TestNumKpsReq:
    def test_enough_valid_unchanged(self):
        """Exactly num_kps valid keypoints: all are kept."""
        kps = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        result = _reshape(num_kps_req(kps, num_kps=3))
        np.testing.assert_array_equal(result[0], [10.0, 20.0])
        np.testing.assert_array_equal(result[2], [50.0, 60.0])

    def test_more_than_required_unchanged(self):
        """More than num_kps valid: all kept, no unnecessary pruning."""
        kps = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = _reshape(num_kps_req(kps, num_kps=2))
        np.testing.assert_array_equal(result[0], [1.0, 2.0])
        np.testing.assert_array_equal(result[2], [5.0, 6.0])

    def test_too_few_valid_clears_all(self):
        """Fewer than num_kps valid → every keypoint is set to (-1, -1)."""
        kps = np.array([[10.0, 20.0], [-1.0, -1.0], [-1.0, -1.0]])
        result = _reshape(num_kps_req(kps, num_kps=2))
        assert np.all(result == -1)

    def test_zero_valid_clears_all(self):
        """No valid keypoints at all → all cleared."""
        kps = np.array([[-1.0, -1.0], [-1.0, -1.0]])
        result = _reshape(num_kps_req(kps, num_kps=1))
        assert np.all(result == -1)

    def test_required_index_invalid_clears_all(self):
        """A required index that is (-1,-1) triggers clearing of everything."""
        kps = np.array([[10.0, 20.0], [30.0, 40.0], [-1.0, -1.0]])
        # Require last keypoint (index 2), which is invalid
        result = _reshape(num_kps_req(kps, num_kps=2, required_indices=[2]))
        assert np.all(result == -1)

    def test_required_index_valid_passes(self):
        """All required indices valid + enough total → no clearing."""
        kps = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        result = _reshape(num_kps_req(kps, num_kps=2, required_indices=[-1]))
        np.testing.assert_array_equal(result[2], [50.0, 60.0])

    def test_negative_required_index(self):
        """Negative indices are resolved modulo len(keypoints)."""
        kps = np.array([[10.0, 20.0], [-1.0, -1.0], [50.0, 60.0]])
        # -2 == index 1, which is invalid → should clear all
        result = _reshape(num_kps_req(kps, num_kps=1, required_indices=[-2]))
        assert np.all(result == -1)

    def test_multiple_required_indices_all_valid(self):
        kps = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = _reshape(num_kps_req(kps, num_kps=2, required_indices=[0, -1]))
        # Both required indices (0 and 2) are valid → no clearing
        np.testing.assert_array_equal(result[0], [1.0, 2.0])

    def test_multiple_required_indices_one_missing(self):
        kps = np.array([[1.0, 2.0], [-1.0, -1.0], [5.0, 6.0]])
        result = _reshape(num_kps_req(kps, num_kps=1, required_indices=[0, 1]))
        # Index 1 is invalid → clear all
        assert np.all(result == -1)


# ---------------------------------------------------------------------------
# kps_postprocessor (full pipeline)
# ---------------------------------------------------------------------------

class TestKpsPostprocessor:
    def test_low_confidence_then_overlap_then_count(self):
        """
        Three keypoints:
          kps[0] → low confidence → filtered out
          kps[1] and kps[2] → close together; kps[1] wins (higher conf)
          After both steps, 1 valid keypoint satisfies num_kps=1.
        """
        kps = np.array([[100.0, 100.0], [200.0, 200.0], [201.0, 200.0]])
        result = _reshape(kps_postprocessor(kps, [0.1, 0.9, 0.5], threshold=0.5, pixel_distance=10, num_kps=1))
        np.testing.assert_array_equal(result[0], INVALID)           # filtered by confidence
        np.testing.assert_array_equal(result[1], [200.0, 200.0])    # kept
        np.testing.assert_array_equal(result[2], INVALID)            # removed by overlap

    def test_not_enough_after_filtering_clears_all(self):
        """
        After confidence filtering, if the surviving count < num_kps,
        the count guard clears everything.
        """
        kps = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        # Only kps[2] survives the confidence filter; num_kps=2 → clears all
        result = _reshape(kps_postprocessor(kps, [0.1, 0.2, 0.9], threshold=0.5, pixel_distance=5, num_kps=2))
        assert np.all(result == -1)

    def test_all_valid_no_overlap_above_count(self):
        """Happy path: all confident, no overlap, above count threshold."""
        kps = np.array([[10.0, 10.0], [100.0, 100.0], [200.0, 200.0]])
        result = _reshape(kps_postprocessor(kps, [0.9, 0.8, 0.7], threshold=0.5, pixel_distance=5, num_kps=3))
        np.testing.assert_array_equal(result[0], [10.0, 10.0])
        np.testing.assert_array_equal(result[1], [100.0, 100.0])
        np.testing.assert_array_equal(result[2], [200.0, 200.0])

    def test_required_index_guard_in_pipeline(self):
        """Pipeline propagates required_indices to num_kps_req correctly."""
        kps = np.array([[10.0, 10.0], [20.0, 20.0], [-1.0, -1.0]])
        # kps[2] is pre-invalid (simulate filtered-out); it is also required
        result = _reshape(kps_postprocessor(
            kps, [0.9, 0.8, 0.0], threshold=0.5, pixel_distance=5,
            num_kps=1, required_indices=[-1]
        ))
        assert np.all(result == -1)
