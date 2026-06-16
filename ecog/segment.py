"""Split a tile into segments — groups of neighbouring, similar pixels.

:func:`segment_image` runs one of two classic superpixel algorithms and returns
a :class:`Segments` object: an integer label image where every pixel carries the
id of the segment it belongs to. :func:`draw_boundaries` renders those segments
for display.
"""

from dataclasses import dataclass

import numpy as np
from skimage.segmentation import slic, felzenszwalb, find_boundaries

# Names shown in the segmentation dropdown.
METHODS = ["SLIC", "Felzenszwalb"]


@dataclass
class Segments:
    """The result of segmentation."""

    labels: np.ndarray   # (height, width) int; each value is a segment id
    count: int           # number of segments (ids run 0 .. count - 1)

    def id_at(self, row: int, col: int) -> int:
        """Return the id of the segment that covers one pixel."""
        return int(self.labels[row, col])

    def mask_of(self, seg_id: int) -> np.ndarray:
        """Return a boolean image that is True inside segment ``seg_id``."""
        return self.labels == seg_id


def segment_image(
    image: np.ndarray,
    method: str = "SLIC",
    *,
    n_segments: int = 10000,
    compactness: float = 10,
) -> Segments:
    """Segment ``image`` with the chosen ``method`` (see :data:`METHODS`).

    ``n_segments`` and ``compactness`` tune SLIC: how many superpixels to aim for
    and how strongly they keep a compact, regular shape. They are ignored by the
    other methods.
    """
    rgb = image[:, :, :3]
    if method == "SLIC":
        # SLIC makes roughly equal-sized, compact superpixels.
        labels = slic(rgb, n_segments=n_segments, compactness=compactness, start_label=0)
    elif method == "Felzenszwalb":
        # Felzenszwalb follows the image edges, so segment sizes vary more.
        # TODO Implement Felzenswalb
        pass
    else:
        # TODO Implement more if you want to
        raise ValueError(f"Unknown method {method!r}. Options: {METHODS}")
    return Segments(labels=labels.astype(int), count=int(labels.max()) + 1)


def draw_boundaries(image: np.ndarray, segments: Segments) -> np.ndarray:
    """Return a copy of ``image`` with thin lines drawn along the segment borders."""
    border = find_boundaries(segments.labels, mode="outer")
    out = image.copy()
    out[border] = (0, 0, 205)   # a strong blue line
    return out
