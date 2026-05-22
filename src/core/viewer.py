import os
from pathlib import Path
import tifffile as tiff
import rasterio
from rasterio.enums import Resampling
import numpy as np


def resample_tif(file_path : Path, scale : int):
        new_file_path = file_path.parent / f"{file_path.stem}_resampled.tif"
        with rasterio.open(file_path) as src:
            out_shape = (src.count, 
                         int(src.height * scale), 
                         int(src.width * scale)
                        )
            resampled_img_array = src.read(
                out_shape = out_shape,
                resampling=Resampling.bilinear
            )
        
            new_transform = src.transform * src.transform.scale(
                (src.width / resampled_img_array.shape[-1]),
                (src.height / resampled_img_array.shape[-2])
            )

            profile = src.profile.copy()
            profile.update({
                "transform" : new_transform,
                "height" : resampled_img_array.shape[1],
                "width" : resampled_img_array.shape[2]
            })

            with rasterio.open(new_file_path, "w", **profile) as dst:
                dst.write(resampled_img_array)
            print("Successfully resampled the tile, so it loads faster. For analysis the original is used")


def load_raster(tile_number : int):
    
    IMAGE_PATH_ROOT = "data"
    dir_name = f"3151{tile_number}_56865"

    # File for processing
    file_name = f"BE_ORTHO_27032011_3151{tile_number}_56865_UTM31N.tif"
    tile_path = os.path.join(IMAGE_PATH_ROOT, dir_name, file_name)

    # File resampled just for viewing
    file_name_resampled = f"BE_ORTHO_27032011_3151{tile_number}_56865_UTM31N_resampled.tif"
    tile_path_resampled = os.path.join(IMAGE_PATH_ROOT, dir_name, file_name_resampled)

    if not os.path.exists(tile_path_resampled):
        resample_tif(Path(tile_path), 0.25)

    resampled = tiff.imread(tile_path_resampled)

    if resampled.shape[0] == 3:
        display_array = np.transpose(resampled, (1,2,0))
    else:
        display_array = resampled

    val_min = np.min(display_array)
    val_max = np.max(display_array)
    norm_array = ((resampled - val_min)/(val_max-val_min) * 255).astype(np.uint8)

    return norm_array
    