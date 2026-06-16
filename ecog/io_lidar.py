from pathlib import Path

import numpy as np
import laspy
import rasterio
from scipy.stats import binned_statistic_2d
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt




# TODO Needs to be implemented, this is currently just to try 
lidar_path = str(Path.cwd() / "data/315130_56865/BE_LIDAR_27032011_315130_56865_UTM31N.las")
dsm_path = lidar_path.replace("LIDAR", "DSM").replace(".las", ".tif")

# OPEN DSM
with rasterio.open(dsm_path) as dsm_src:
    dsm = dsm_src.read(1).astype(float)
    transform = dsm_src.transform
    crs = dsm_src.crs
    height, width = dsm_src.height, dsm_src.width
    bounds = dsm_src.bounds

# Read in lidar    
las = laspy.read(lidar_path)

# check crs

print("LAS CRS: ", las.header.parse_crs())
print("DSM CRS: ", crs)


# 1. Alle Punkte nutzen (da keine Filter möglich sind)
x = las.x
y = las.y
z = las.z

# 2. Rastern mit robuster Statistik gegen Rauschen
result = binned_statistic_2d(
    y, x, z,
    statistic='max',  # Nutzt den mittleren Wert der Zelle, ignoriert hohe Ausreißer
    bins=[height, width],
    range=[[bounds.bottom, bounds.top], [bounds.left, bounds.right]]
)

dsm_lidar = result.statistic

dsm_lidar = np.flipud(dsm_lidar)

ndsm = dsm_lidar - dsm
ndsm = np.clip(ndsm, 0, None)

ndsm = np.where(np.isnan(ndsm), 0.0, ndsm)


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# DSM
im0 = axes[0].imshow(dsm_lidar, cmap='rainbow')
axes[0].set_title('DSM (absolute Höhe)')
plt.colorbar(im0, ax=axes[0], label='m')

# DTM
im1 = axes[1].imshow(dsm, cmap='rainbow')
axes[1].set_title('DTM (Bodenhöhe)')
plt.colorbar(im1, ax=axes[1], label='m')

# nDSM
im2 = axes[2].imshow(ndsm, cmap='viridis', vmin=0, vmax=np.nanmax(ndsm))
axes[2].set_title('nDSM (Objekthöhe)')
plt.colorbar(im2, ax=axes[2], label='m')

plt.tight_layout()
plt.savefig("comparison.png", dpi=150)
plt.show()

