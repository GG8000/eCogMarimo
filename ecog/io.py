"""Load an aerial ortho tile and resample it to a workable size.

The one function you need is :func:`load_tile`. It returns a :class:`Tile`:
the RGB image as a ``uint8`` array, plus the affine transform that maps pixel
positions to real-world coordinates.
"""

from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
import rasterio
from scipy.stats import binned_statistic_2d
from rasterio.enums import Resampling
from rasterio.warp import reproject



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
    ndsm: np.ndarray    # (height, width) float32; height of moving/temporary objects (containers, cranes, ...) above the clean DSM
    transform: object   # rasterio affine: (col, row) -> (x, y) in world coordinates
    crs: object         # coordinate reference system of the tile

def load_tile(folder: str, size: int = 1500, data_root: Path = DATA_ROOT) -> Tile:
    """Read one tile's ortho photo and resample it to ``size`` x ``size`` pixels.

    We read the full 10000x10000 ortho so segmentation and classification can run
    at a high resolution; rasterio reads it *decimated* straight to ``size``, so
    this stays fast even though the file is large. The smaller ``_resampled.tif``
    (2500x2500) is used only as a fallback if the full ortho is missing. The
    image is resampled to ``size`` and its colours are stretched for clarity.
    """
    stem = f"BE_ORTHO_27032011_{folder}_UTM31N"
    full = data_root / folder / f"{stem}.tif"
    path = full if full.exists() else data_root / folder / f"{stem}_resampled.tif"
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

    ndsm = load_ndsm(
        folder=folder,
        size=size,
        transform=transform,
        crs=crs
    )
    
    # rasterio gives (band, row, col); numpy and images want (row, col, band).
    image = np.transpose(bands, (1, 2, 0))
    return Tile(image=stretch_contrast(image), ndsm=ndsm, transform=transform, crs=crs)

def load_ndsm(
    folder: str,
    size: int,
    transform: object,
    crs: object,
    min_height: float = 1.0,
) -> np.ndarray:
    """Height of moving/temporary objects (containers, cranes, ...) on the tile.

    The raw LiDAR cloud was captured *with* those objects in place; the provided
    DSM has them removed. Rasterising the cloud to a surface and subtracting the
    clean DSM therefore leaves exactly their height above the static surface,
    which we then warp onto the tile's pixel grid.

    ``min_height`` (metres) suppresses noise and sub-pixel edge artefacts that
    arise from differencing two surfaces: anything lower is set to 0 so only real
    objects remain. Set it to 0 to keep every positive difference.
    """
    stem_lidar = f"BE_LIDAR_27032011_{folder}_UTM31N"
    stem_dsm = f"BE_DSM_27032011_{folder}_UTM31N"
    full_lidar = DATA_ROOT / folder / f"{stem_lidar}.las"
    full_dsm = DATA_ROOT / folder / f"{stem_dsm}.tif"
    
    if not full_lidar.exists():
        raise FileNotFoundError(f"Lidar data not found: {full_lidar}")
    if not full_dsm.exists():
        raise FileNotFoundError(f"DSM (without moving objects) data not found: {full_dsm}")
    
    # Read in lidar data
    las = laspy.read(full_lidar)
    # Read in dsm data
    with rasterio.open(full_dsm) as dsm_src:
        dsm = dsm_src.read(1).astype(float)
        bounds = dsm_src.bounds
        height, width = dsm_src.height, dsm_src.width
        src_transform = dsm_src.transform
        src_crs = dsm_src.crs

    # 2. Rastern mit robuster Statistik gegen Rauschen
    result = binned_statistic_2d(
        las.y, las.x, las.z,
        statistic='median',  # Nutzt den mittleren Wert der Zelle, ignoriert hohe Ausreißer
        bins=[height, width],
        range=[[bounds.bottom, bounds.top], [bounds.left, bounds.right]]
    )

    dsm_lidar = result.statistic

    dsm_lidar = np.flipud(dsm_lidar)

    # Difference of the two surfaces. Empty LiDAR cells come back as NaN; fill
    # them here, *before* warping, so the bilinear reproject does not smear the
    # holes into their neighbours.
    object_height = dsm_lidar - dsm
    object_height = np.where(np.isnan(object_height), 0.0, object_height).astype(np.float32)

    # Warp from the DSM's grid onto the tile's pixel grid so the result lines up
    # with `image` and the segment label map.
    aligned = np.zeros((size, size), dtype=np.float32)
    reproject(
        source=object_height,
        destination=aligned,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=crs,
        resampling=Resampling.bilinear,
    )

    # Keep only positive height (present in the LiDAR but not the clean DSM),
    # then drop everything below `min_height` so building-edge noise from the
    # differencing does not masquerade as an object.
    aligned = np.clip(aligned, 0, None)
    aligned[aligned < min_height] = 0.0
    return aligned

    
    

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
