# Testing

#### Note: Tests are created based on instructions by Claude AI with some modifications.

Run all tests with:
```
poetry run pytest
```

84 tests across 4 files. All tests run on CPU with no dataset or pretrained weights required.

---

## Testing of heatmap (17 tests)
Tests `create_heatmap` and `extract_keypoints` in isolation and as a combined round-trip.
Core invariant: `create_heatmap(kps) → extract_keypoints()` must recover the original keypoints within ±1 pixel.

### TestCreateHeatmap
| Test | Description |
|---|---|
| `test_output_shape` | Output tensor has shape `[B, K, H, W]` |
| `test_values_in_zero_one` | All heatmap values lie in `[0, 1]` |
| `test_peak_at_correct_pixel` | Argmax of each channel falls on the specified keypoint pixel |
| `test_corner_top_left` | Keypoint `(0, 0)` places the peak at pixel `(0, 0)` |
| `test_corner_bottom_right` | Keypoint `(1, 1)` places the peak at pixel `(W-1, H-1)` |
| `test_negative_keypoint_produces_zero_heatmap` | Keypoints with `x < 0` or `y < 0` produce an all-zero channel |
| `test_multiple_keypoints_independent_channels` | Each keypoint maps to its own channel with no cross-channel bleed |
| `test_batch_dimension_independent` | Each batch element is processed independently, producing distinct peaks |

### TestExtractKeypoints
| Test | Description |
|---|---|
| `test_output_shape` | Output tensor has shape `[B, K*2]` |
| `test_values_in_zero_one` | Extracted normalised coordinates lie in `[0, 1]` |
| `test_return_max_values_shape` | With `return_max_values=True`, returns `(keypoints, list[list[float]])` of correct sizes |
| `test_max_value_matches_heatmap_peak` | Reported confidence equals the true channel maximum |
| `test_single_hot_pixel` | A single lit pixel is recovered to the exact normalised coordinate |

### TestHeatmapRoundTrip
| Test | Description |
|---|---|
| `test_single_center_keypoint` | Center keypoint `(0.5, 0.5)` survives the round-trip within 1-pixel tolerance |
| `test_two_keypoints_per_image` | Two keypoints in a single image are both recovered correctly |
| `test_batch_round_trip` | All keypoints across a batch of 4 images are recovered within tolerance |
| `test_corner_keypoints_round_trip` | All four image corners survive the round-trip |

---

## Testing of post-processing (27 tests)

Tests the three filtering steps that clean up raw heatmap predictions before use.

### TestFilterLowProbabilities
| Test | Description |
|---|---|
| `test_below_threshold_becomes_invalid` | Keypoints with confidence below threshold are set to `(-1, -1)` |
| `test_above_threshold_unchanged` | Keypoints at or above threshold are left untouched |
| `test_all_below_threshold` | All keypoints below threshold → all become `(-1, -1)` |
| `test_exactly_at_threshold_is_kept` | A confidence equal to the threshold is kept (strict `<`, not `<=`) |
| `test_flat_input_format` | Accepts flat `[x1, y1, x2, y2, ...]` format in addition to `(N, 2)` |
| `test_output_is_copy_not_in_place` | The original array is not mutated |

### TestOverlappingKps
| Test | Description |
|---|---|
| `test_lower_confidence_of_pair_is_removed` | When two keypoints are closer than `pixel_distance`, the weaker one is removed |
| `test_higher_confidence_wins_when_reversed` | Same pair with swapped confidences → the other one is removed |
| `test_far_apart_keypoints_both_kept` | Keypoints beyond `pixel_distance` are both preserved |
| `test_exactly_at_distance_threshold_both_kept` | Distance exactly equal to threshold is not considered overlapping (strict `<`) |
| `test_existing_invalid_keypoints_ignored` | Pre-existing `(-1, -1)` entries are excluded from overlap checks |
| `test_single_keypoint_unchanged` | A single keypoint cannot overlap with itself |
| `test_all_invalid_returns_all_invalid` | All-invalid input passes through unchanged |
| `test_three_keypoints_chain_overlap` | Three mutually close keypoints: only the highest-confidence one survives |

### TestNumKpsReq
| Test | Description |
|---|---|
| `test_enough_valid_unchanged` | Exactly `num_kps` valid keypoints → nothing is cleared |
| `test_more_than_required_unchanged` | More than `num_kps` valid → nothing is unnecessarily pruned |
| `test_too_few_valid_clears_all` | Fewer than `num_kps` valid → all keypoints set to `(-1, -1)` |
| `test_zero_valid_clears_all` | No valid keypoints → all cleared |
| `test_required_index_invalid_clears_all` | A required index that is `(-1, -1)` triggers clearing of all keypoints |
| `test_required_index_valid_passes` | All required indices valid + enough total count → no clearing |
| `test_negative_required_index` | Negative indices are resolved modulo `len(keypoints)` |
| `test_multiple_required_indices_all_valid` | Multiple required indices, all valid → no clearing |
| `test_multiple_required_indices_one_missing` | Multiple required indices, one invalid → all cleared |

### TestKpsPostprocessor
| Test | Description |
|---|---|
| `test_low_confidence_then_overlap_then_count` | Full pipeline: filter by confidence, remove overlaps, then count check — in that order |
| `test_not_enough_after_filtering_clears_all` | After filtering, if the surviving count falls below `num_kps`, everything is cleared |
| `test_all_valid_no_overlap_above_count` | Happy path: all confident, no overlap, count satisfied → all kept |
| `test_required_index_guard_in_pipeline` | `required_indices` is correctly propagated through the full pipeline |

---

## Testing of metrics (25 tests)
Tests `compute_pck`, `compute_mpjpe`, `compute_loss`, and the RCNN keypoint loss helper.

### TestComputePCK
| Test | Description |
|---|---|
| `test_perfect_predictions_give_one` | Identical preds and targets → PCK = 1.0 |
| `test_far_predictions_give_zero` | Predictions far beyond the threshold → PCK = 0.0 |
| `test_half_correct_gives_half_pck` | Exactly half the keypoints within threshold → PCK = 0.5 |
| `test_per_keypoint_shape` | `pck_per_keypoint` has shape `(K,)` |
| `test_num_total` | `num_total` equals `B × K` |
| `test_flat_input_accepted` | Flat `[B, K*2]` tensors are accepted and reshaped internally |
| `test_threshold_sensitivity` | A tighter threshold yields equal or lower PCK |
| `test_pck_regression_gate_near_perfect_predictions` | **Regression gate**: PCK@0.05 ≥ 0.95 with ±2 px noise — catches regressions in the metric itself |
| `test_pck_regression_gate_medium_noise` | **Regression gate**: PCK@0.1 ≥ 0.90 with ±10 px noise |

### TestComputeMPJPE
| Test | Description |
|---|---|
| `test_zero_error` | Identical preds and targets → MPJPE = 0 |
| `test_known_euclidean_distance` | 3-4-5 triangle offset → MPJPE = exactly 5.0 |
| `test_per_keypoint_shape` | `mpjpe_per_keypoint` has shape `(K,)` |
| `test_std_is_non_negative` | Standard deviation of per-keypoint errors is ≥ 0 |
| `test_flat_input_accepted` | Flat `[B, K*2]` tensors are accepted and reshaped internally |
| `test_symmetric` | `MPJPE(A, B) == MPJPE(B, A)` |

### TestComputeLossHeatmap
| Test | Description |
|---|---|
| `test_perfect_preds_lower_loss_than_random` | Heatmaps generated from the target keypoints produce lower loss than random predictions |
| `test_returns_scalar_tensor` | Loss is a scalar tensor (shape `()`) |
| `test_loss_is_non_negative` | BCEWithLogitsLoss is always ≥ 0 |
| `test_unknown_model_raises_value_error` | An unrecognised `model_name` raises `ValueError` |

### TestCalculateKeypointLoss
| Test | Description |
|---|---|
| `test_zero_loss_when_preds_match_targets` | Predictions matching targets exactly → loss = 0 |
| `test_non_zero_loss_when_preds_differ` | Different predictions produce positive loss |
| `test_occluded_keypoints_excluded_from_loss` | Keypoints with `visibility = 0` are masked out and do not affect the loss |
| `test_empty_detection_emits_runtime_warning` | A batch with no detections triggers `RuntimeWarning` |
| `test_empty_detection_returns_zero` | Despite no detections, the function returns a finite `0.0` |
| `test_batch_averages_over_valid_images` | Loss is averaged only over images that have detections; empty images are excluded |

---

## Testing of models (15 tests)
All model tests run on CPU with `pretrained=False` — no network downloads required.
KeypointRCNN is excluded because it hardcodes `weights='DEFAULT'`.

### TestResNetKeypoint
| Test | Description |
|---|---|
| `test_output_shape_batch_2` | Output shape is `[2, num_kps * 2]` for a batch of 2 |
| `test_output_shape_batch_1` | Output shape is `[1, num_kps * 2]` for a single image |
| `test_output_is_finite` | Output contains no NaN or Inf values |
| `test_output_dtype_float32` | Output dtype is `float32` |
| `test_eval_mode_consistent_across_batch_sizes` | The same image repeated in a batch produces identical per-image outputs in eval mode |
| `test_gradient_flows` | A gradient flows back through the entire model |
| `test_different_spatial_inputs_work` | `AdaptiveAvgPool2d` allows any spatial input size |

### TestResNetHeatmap
| Test | Description |
|---|---|
| `test_output_channels_equals_num_kps` | Output has exactly `num_kps` channels |
| `test_output_spatial_larger_than_backbone_stride` | The heatmap head upsamples beyond the 1/32 backbone stride |
| `test_output_batch_dimension` | Batch dimension is preserved |
| `test_output_is_finite` | Output contains no NaN or Inf values |
| `test_output_dtype_float32` | Output dtype is `float32` |
| `test_eval_mode_consistent_across_batch_sizes` | The same image repeated in a batch produces identical per-image outputs in eval mode |
| `test_gradient_flows` | A gradient flows back through the entire model |
| `test_num_kps_respected` | Instantiating with `num_kps ∈ {4, 7, 14}` produces the correct number of output channels |