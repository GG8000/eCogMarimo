import numpy as np, laspy
import rioxarray


# TODO Needs to be implemented, this is currently just to try 

path = "../eCogMarimo/data/315130_56865/BE_LIDAR_27032011_315130_56865_UTM31N.las"
dtm_path = "../eCogMarimo/data/315130_56865/BE_DSM_27032011_315130_56865_UTM31N.tif"

las = laspy.read(path)                      # loads all ~20M points (~1-2 GB RAM)
print("points:", las.header.point_count)
print("dims:", list(las.point_format.dimension_names))
print("mins:", las.header.mins, "maxs:", las.header.maxs)

# real-world coordinates (already scaled/offset for you)
x, y, z = las.x, las.y, las.z
print("Z stats:", z.min(), np.median(z), z.max())

# intensity, return structure, and the standard classification codes
# (2 = ground, 5 = high vegetation, 6 = building, ...)
print("classes:", np.unique(las.classification, return_counts=True))
r, g, b = las.red, las.green, las.blue      # 16-bit per-point colour (PDRF 3)

# Read in dsm
dsm = rioxarray.open_rasterio(dtm_path)
print(dsm)
