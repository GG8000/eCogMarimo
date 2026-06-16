"""Describe each segment with a few numbers the classifier can learn from.

Each segment is described by the mean and standard deviation of its red, green
and blue pixels plus its object height (the nDSM) — eight numbers in total.
:func:`all_features` builds that matrix for *every* segment in one pass, and
:func:`training_table` simply picks out the rows for the segments the user
labelled. Both read from the same matrix, so a segment's training row is always
identical to the row it is later classified on.
"""

import numpy as np

from ecog.io import Tile
from ecog.segment import Segments


def all_features(tile: Tile, segments: Segments) -> np.ndarray:
    """Return the feature matrix for every segment, computed in one fast pass.

    For each of the four channels — red, green, blue and object height (nDSM) —
    we want its mean and standard deviation inside every segment. Rather than
    loop over the segments, we use ``np.bincount``, which sums values grouped by
    segment id in one go; then, per channel, ``mean = sum / pixel_count`` and
    ``std = sqrt(mean_of_squares - mean**2)``. Each row is one segment:
    ``[meanR, meanG, meanB, meanHeight, stdR, stdG, stdB, stdHeight]``.
    """
    ids = segments.labels.ravel()
    rgb = tile.image[:, :, :3].reshape(-1, 3).astype(np.float64)
    height = tile.ndsm.ravel().astype(np.float64)
    channels = [rgb[:, 0], rgb[:, 1], rgb[:, 2], height]
    pixel_count = np.maximum(np.bincount(ids, minlength=segments.count), 1)

    means, stds = [], []
    for values in channels:
        mean = np.bincount(ids, weights=values, minlength=segments.count) / pixel_count
        mean_sq = np.bincount(ids, weights=values * values, minlength=segments.count) / pixel_count
        means.append(mean)
        stds.append(np.sqrt(np.clip(mean_sq - mean * mean, 0, None)))

    return np.column_stack(means + stds).astype(np.float32)


def training_table(tile: Tile, segments: Segments, labels: dict):
    """Build the training data from the user's labels.

    ``labels`` maps a segment id to a class name, e.g. ``{12: "Water"}``. We take
    the rows of :func:`all_features` for the labelled ids, so a training row is
    exactly the row that segment is later classified on. Returns ``X`` (one
    feature row per labelled segment) and ``y`` (the class names in the same
    order).
    """
    features = all_features(tile, segments)
    X, y = [], []
    for seg_id, class_name in labels.items():
        X.append(features[int(seg_id)])
        y.append(class_name)
    return np.array(X), y
