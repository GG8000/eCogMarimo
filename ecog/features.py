"""Describe each segment with a few numbers the classifier can learn from.

A segment is described by the mean and standard deviation of its red, green and
blue pixels — six numbers in total. :func:`training_table` builds the
``(X, y)`` pair from the segments the user labelled; :func:`all_features` builds
the feature matrix for *every* segment, which is what we feed in to predict.
"""

import numpy as np

from ecog.segment import Segments


def segment_features(image: np.ndarray, segments: Segments, seg_id: int) -> np.ndarray:
    """Return the six features (RGB mean + RGB std) of a single segment."""
    pixels = image[segments.mask_of(seg_id)][:, :3]   # (n_pixels, 3)
    if pixels.size == 0:
        return np.zeros(6, dtype=np.float32)
    return np.concatenate([pixels.mean(axis=0), pixels.std(axis=0)]).astype(np.float32)


def training_table(image: np.ndarray, segments: Segments, labels: dict):
    """Build the training data from the user's labels.

    ``labels`` maps a segment id to a class name, e.g. ``{12: "Water"}``.
    Returns ``X`` (one feature row per labelled segment) and ``y`` (the class
    names in the same order).
    """
    X, y = [], []
    for seg_id, class_name in labels.items():
        X.append(segment_features(image, segments, int(seg_id)))
        y.append(class_name)
    return np.array(X), y


def all_features(image: np.ndarray, segments: Segments) -> np.ndarray:
    """Return the feature matrix for every segment, computed in one fast pass.

    We want the mean and std of R, G and B inside each of the segments.
    Rather than loop over every segment, we use ``np.bincount``, which adds up
    values grouped by segment id in one go. Then, per channel,s
    ``mean = sum / pixel_count`` and ``std = sqrt(mean_of_squares - mean**2)``.
    The result has one row per segment id: ``[meanR, meanG, meanB, stdR, stdG, stdB]``.
    """
    ids = segments.labels.ravel()
    rgb = image[:, :, :3].reshape(-1, 3).astype(np.float64)
    pixel_count = np.maximum(np.bincount(ids, minlength=segments.count), 1)

    means, stds = [], []
    for channel in range(3):
        values = rgb[:, channel]
        mean = np.bincount(ids, weights=values, minlength=segments.count) / pixel_count
        mean_sq = np.bincount(ids, weights=values * values, minlength=segments.count) / pixel_count
        means.append(mean)
        stds.append(np.sqrt(np.clip(mean_sq - mean * mean, 0, None)))

    return np.column_stack(means + stds).astype(np.float32)
