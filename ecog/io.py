"""Load an aerial ortho tile and resample it to a workable size.

The one function you need is :func:`load_tile`. It returns a :class:`Tile`:
the RGB image as a ``uint8`` array, plus the affine transform that maps pixel
positions to real-world coordinates.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling



# Where the raster tiles live. This simple app reuses the data folder of the
# original eCogMarimo project, which sits next to this one.
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

# Name shown in the dropdown -> the folder that holds that tile.
TILES = {
    "Tile 30": "315130_56865",
    "Tile 35": "315135_56865",
    "Tile 40": "315140_56865",
}


@dataclass
class Tile:
    """One loaded RGB tile."""

    image: np.ndarray   # (height, width, 3) uint8, contrast-stretched for display
    transform: object   # rasterio affine: (col, row) -> (x, y) in world coordinates
    crs: object         # coordinate reference system of the tile


def load_tile(folder: str, size: int = 1500, data_root: Path = DATA_ROOT) -> Tile:
    """Read one tile's ortho photo and resample it to ``size`` x ``size`` pixels.

    We read the small ``_resampled.tif`` (2500x2500) rather than the full
    10000x10000 ortho, so loading takes a fraction of a second. The image is
    resampled to ``size`` and its colours are stretched for a clearer picture.
    """
    path = data_root / folder / f"BE_ORTHO_27032011_{folder}_UTM31N_resampled.tif"
    if not path.exists():
        raise FileNotFoundError(f"Tile image not found: {path}")

    with rasterio.open(path) as src:
        # Read the first three bands (red, green, blue) straight into the target size.
        bands = src.read(
            indexes=[1, 2, 3],
            out_shape=(3, size, size),
            resampling=Resampling.bilinear,
        )
        # Scale the transform so it still matches the smaller pixel grid.
        transform = src.transform * src.transform.scale(src.width / size, src.height / size)
        crs = src.crs

    # rasterio gives (band, row, col); numpy and images want (row, col, band).
    image = np.transpose(bands, (1, 2, 0))
    return Tile(image=stretch_contrast(image), transform=transform, crs=crs)


def stretch_contrast(image: np.ndarray) -> np.ndarray:
    """Stretch each colour channel to the full 0-255 range for a clearer image.

    Each band is stretched on its own between its 2nd and 98th percentile, so a
    few very bright or very dark pixels do not wash out the rest of the picture.
    """
    out = np.zeros_like(image, dtype=np.uint8)
    for channel in range(image.shape[2]):
        band = image[:, :, channel].astype(np.float32)
        low, high = np.percentile(band, (2, 98))
        if high - low < 1e-6:
            continue  # flat band: nothing to stretch, leave it black
        scaled = (np.clip(band, low, high) - low) / (high - low) * 255
        out[:, :, channel] = scaled.astype(np.uint8)
    return out
