"""Split a tile into segments — groups of neighbouring, similar pixels.

:func:`segment_image` runs one of two classic superpixel algorithms and returns
a :class:`Segments` object: an integer label image where every pixel carries the
id of the segment it belongs to. :func:`downscale_segments` shrinks that label
map for display; the segment borders themselves are drawn in the browser by the
sampler widget, straight from the id map.
"""

from dataclasses import dataclass

import numpy as np
from skimage.segmentation import slic, felzenszwalb

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



# When using more seg methods use **kwargs, it takes all additional named arguments 
def segment_image(
    image: np.ndarray,
    object_heigth: np.ndarray,
    method: str = "SLIC",
    *, # Ensures every arguments afterwards needs to be passed with the name
    n_segments: int = 10000,
    compactness: float = 10,
) -> Segments:
    """Segment ``image`` with the chosen ``method`` (see :data:`METHODS`).

    ``n_segments`` and ``compactness`` tune SLIC: how many superpixels to aim for
    and how strongly they keep a compact, regular shape. They are ignored by the
    other methods.
    """
    rgb = image[:, :, :3].astype(np.float64)
    
    if method == "SLIC":
        # SLIC makes roughly equal-sized, compact superpixels.
        # n_segments = kwargs.get("n_segments", 10000)
        # compactness = kwargs.get("compactness", 10)
        labels = slic(rgb, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=-1)
    elif method == "Felzenszwalb":
        # Felzenszwalb follows the image edges, so segment sizes vary more.
        # scale = kwargs.get("scale", 1.0)
        # sigma = kwargs.get("sigma", 0.8)
        # TODO Implement Felzenswalb
        raise NotImplementedError("Felzenszwalb not implemented yet")
    else:
        # TODO Implement more if you want to
        raise ValueError(f"Unknown method {method!r}. Options: {METHODS}")
    return Segments(labels=labels.astype(int), count=int(labels.max()) + 1)


def downscale_segments(segments: Segments, size: int) -> Segments:
    """Return a ``size`` x ``size`` copy of the label map for display.

    Segmentation runs on the full-resolution image, but the clickable widget
    needs a much smaller label map (a high-resolution one would be far too big to
    embed in the page). We pick representative pixels with nearest-neighbour
    sampling so every id stays an exact, valid segment id.
    ``count`` is unchanged: the labels collected here index the same segments the
    classifier trains on. Segments too small to survive the shrink simply cannot
    be clicked, but are still classified.
    """
    rows = np.linspace(0, segments.labels.shape[0] - 1, size).round().astype(int)
    cols = np.linspace(0, segments.labels.shape[1] - 1, size).round().astype(int)
    return Segments(labels=segments.labels[np.ix_(rows, cols)], count=segments.count)
