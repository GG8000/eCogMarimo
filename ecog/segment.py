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


def segment_image(
    image: np.ndarray,
    object_heigth: np.ndarray,
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
    # Stack colour and height into one (height, width, 4) image so segmentation
    # can use both. The height channel is scaled to roughly the 0..255 range of
    # the colour channels so it carries comparable weight in the distance SLIC
    # minimises (otherwise metre-scale heights would be drowned out by colour).
    rgb = image[:, :, :3].astype(np.float64)
    
    if method == "SLIC":
        # SLIC makes roughly equal-sized, compact superpixels.
        labels = slic(rgb, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=-1)
    elif method == "Felzenszwalb":
        # Felzenszwalb follows the image edges, so segment sizes vary more.
        # TODO Implement Felzenswalb
        pass
    else:
        # TODO Implement more if you want to
        raise ValueError(f"Unknown method {method!r}. Options: {METHODS}")
    return Segments(labels=labels.astype(int), count=int(labels.max()) + 1)


def downscale_segments(segments: Segments, size: int) -> Segments:
    """Return a ``size`` x ``size`` copy of the label map for display.

    Segmentation runs on the full-resolution image, but the clickable widget
    needs a much smaller label map (a high-resolution one would be far too big to
    embed in the page). We pick representative pixels with nearest-neighbour
    sampling — never averaging — so every id stays an exact, valid segment id.
    ``count`` is unchanged: the labels collected here index the same segments the
    classifier trains on. Segments too small to survive the shrink simply cannot
    be clicked, but are still classified.
    """
    rows = np.linspace(0, segments.labels.shape[0] - 1, size).round().astype(int)
    cols = np.linspace(0, segments.labels.shape[1] - 1, size).round().astype(int)
    return Segments(labels=segments.labels[np.ix_(rows, cols)], count=segments.count)
