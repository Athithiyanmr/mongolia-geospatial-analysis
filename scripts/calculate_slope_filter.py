from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ============================================================
# INPUTS
# ============================================================

LULC_RASTER = Path(
    "outputs/rasters/LULC_2025_AOI_EPSG32648_filled.tif"
)

SLOPE_RASTER = Path(
    "data/raw/terrain/aoi_slope_30m.tif"
)

OUTPUT_RASTER = Path(
    "outputs/rasters/grassland_dry_steppe_slope_lt8_filled.tif"
)

# ============================================================

print("\nReading LULC...")

with rasterio.open(LULC_RASTER) as lulc_src:

    lulc = lulc_src.read(1)

    profile = lulc_src.profile.copy()

    transform = lulc_src.transform
    crs = lulc_src.crs
    width = lulc_src.width
    height = lulc_src.height

    lulc_nodata = lulc_src.nodata

print("Reading Slope...")

with rasterio.open(SLOPE_RASTER) as slope_src:

    slope = np.full(
        (height, width),
        np.nan,
        dtype=np.float32,
    )

    reproject(
        source=rasterio.band(slope_src, 1),
        destination=slope,

        src_transform=slope_src.transform,
        src_crs=slope_src.crs,

        dst_transform=transform,
        dst_crs=crs,

        dst_width=width,
        dst_height=height,

        src_nodata=slope_src.nodata,
        dst_nodata=np.nan,

        resampling=Resampling.bilinear,
    )

print("Filtering landcover...")

land_mask = np.isin(
    lulc,
    [2, 3],
)

print("Filtering slope...")

slope_mask = (
    (~np.isnan(slope))
    &
    (slope <= 8)
)

final_mask = (
    land_mask
    &
    slope_mask
)

output = np.full(
    lulc.shape,
    lulc_nodata,
    dtype=lulc.dtype,
)

output[final_mask] = lulc[final_mask]

profile.update(
    compress="lzw",
    tiled=True,
)

OUTPUT_RASTER.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with rasterio.open(
    OUTPUT_RASTER,
    "w",
    **profile,
) as dst:

    dst.write(output, 1)

print("\nFinished")
print("-" * 50)
print(f"Grassland pixels : {(lulc == 2).sum():,}")
print(f"Dry Steppe pixels: {(lulc == 3).sum():,}")
print(f"Pixels after slope filter: {final_mask.sum():,}")
print(f"\nOutput saved to:\n{OUTPUT_RASTER}")