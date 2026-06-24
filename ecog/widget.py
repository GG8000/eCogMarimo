"""The interactive sampler: an image widget you label by clicking segments.

This is the only front-end code in the project. The widget is deliberately
self-contained: Python hands it the picture, a hidden map of which segment each
pixel belongs to, and the list of classes. From then on it does everything in
the browser — picking a class, tinting clicked segments, zooming and panning —
and only the collected labels (segment id -> class name) are synced back to
Python. Because labelling never touches Python, the widget is never rebuilt
mid-session: clicks are instant and the zoom level stays put.
"""

import base64
from io import BytesIO

import anywidget
import traitlets
import numpy as np
import marimo as mo
import matplotlib
from PIL import Image


def image_bytes(image: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 image as PNG bytes."""
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def downscale(image: np.ndarray, size: int) -> np.ndarray:
    """Shrink an (H, W, 3) uint8 photo to ``size`` x ``size`` for display.

    Computation runs on the full-resolution image, but what we embed in the page
    (the RGB view and the clickable widget) must stay small. A smooth (bilinear)
    resize is right for a photo; the segment-id map is shrunk separately with
    :func:`ecog.segment.downscale_segments`, which must keep ids exact.
    """
    return np.asarray(Image.fromarray(image).resize((size, size), Image.BILINEAR))


def colorize_height(ndsm: np.ndarray, max_height: float = 10.0) -> np.ndarray:
    """Turn the object-height map (metres) into a colour image for display.

    ``ndsm`` is a float array of heights; ``mo.image`` needs a uint8 RGB
    picture. We clip at ``max_height`` metres, scale to 0..1 and run it through
    the viridis colour map, so ground level (0 m) reads as dark blue and taller
    objects shade towards green and yellow. Raise ``max_height`` if the tallest
    objects all look the same bright yellow (the scale has saturated).
    """
    if np.nanmax(ndsm): max_height = np.nanmax(ndsm)
    
    norm = np.clip(ndsm, 0, max_height) / max_height
    rgb = matplotlib.colormaps["viridis"](norm)[:, :, :3]   # drop the alpha channel
    return (rgb * 255).astype(np.uint8)


def png_data_url(image: np.ndarray) -> str:
    """Encode an (H, W, 3) uint8 image as a base64 PNG data URL (for ``mo.image``).

    Passing this string to ``mo.image`` embeds the picture straight into the
    HTML. Passing raw bytes instead makes marimo register a *virtual file*
    served at its own URL — and those are dropped when a cell re-runs, so in
    ``marimo run`` the browser can ask for one that no longer exists and the
    server logs a 404 ("Virtual file not found"). A data URL has nothing to
    fetch, so it cannot go stale.
    """
    return "data:image/png;base64," + base64.b64encode(image_bytes(image)).decode()


def data_url(image: np.ndarray) -> str:
    """Encode an (H, W, 3) uint8 image as a base64 JPEG data URL so not the whole image need to be fetched.

    Used for photographic images (the clickable widget and the RGB view): JPEG
    keeps them small, whereas a base64 PNG of a 1500x1500 photo is ~5 MB and
    overflows marimo's per-cell output limit. Use :func:`png_data_url` instead
    for flat-colour images like the classification map, where exact colours
    matter and PNG stays small.
    """
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def id_map_url(labels: np.ndarray) -> str:
    """Encode the segment-id image as a lossless PNG data URL.

    Each segment id is packed into a pixel's red and green channels
    (``id = red + green * 256``), so the browser can look up which segment any
    pixel belongs to. This supports up to 65 535 segments. PNG is essential here, 
    the ids must survive exactly.
    """
    ids = labels.astype(np.uint32)
    rgb = np.zeros(ids.shape + (3,), dtype=np.uint8)
    # Pack the 16-bit segment id into two 8-bit channels so the browser can
    # reconstruct the id exactly. The low byte (bits 0-7) goes into the red
    # channel and the high byte (bits 8-15) into the green channel.
    # id = red + green * 256
    rgb[:, :, 0] = ids & 0xFF
    rgb[:, :, 1] = (ids >> 8) & 0xFF
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class SegmentSampler(anywidget.AnyWidget):
    """A clickable, zoomable image for labelling segments.

    Read ``.value["labels"]`` from another cell to get ``{segment_id: class}``
    (ids are strings, as JSON requires). Everything else is input the widget
    needs and never changes after it is built. 
    """
    
    # traitlets stores the metadata and the states
    image_src = traitlets.Unicode("").tag(sync=True)   # the picture, as a data URL
    seg_src = traitlets.Unicode("").tag(sync=True)      # the packed segment-id map
    classes = traitlets.Dict().tag(sync=True)           # class name -> [r, g, b]
    width = traitlets.Unicode("100%").tag(sync=True)    # CSS max-width of the widget
    labels = traitlets.Dict().tag(sync=True)            # segment id (str) -> class name

    _esm = "ecog/widget.js"


def segment_sampler(image: np.ndarray, segments, classes: dict, width: str = "100%"):
    """Build the interactive sampler for one segmented tile.

    ``image`` is the picture to show (with segment borders already drawn);
    ``segments`` carries the per-pixel segment ids; ``classes`` maps each class
    name to its RGB colour. Read the collected labels from ``.value["labels"]``.
    """
    return mo.ui.anywidget(SegmentSampler(
        image_src=data_url(image),
        seg_src=id_map_url(segments.labels),
        classes={name: list(colour) for name, colour in classes.items()},
        width=width,
    ))
