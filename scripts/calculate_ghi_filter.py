from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ============================================================
# INPUTS
# ============================================================

# Output from slope filtering
INPUT_RASTER = Path(
    "outputs/rasters/grassland_dry_steppe_slope_lt8.tif"
)

# GHI Raster
GHI_RASTER = Path(
    "data/raw/GHI/GHI.tif"
)

# Output raster
OUTPUT_RASTER = Path(
    "outputs/rasters/grassland_dry_steppe_slope_lt8_GHI45.tif"
)

# Minimum GHI threshold
GHI_THRESHOLD = 4.5

# ============================================================
# READ FILTERED LAND COVER
# ============================================================

print("\nReading filtered land cover raster...")

with rasterio.open(INPUT_RASTER) as src:

    landcover = src.read(1)

    profile = src.profile.copy()

    transform = src.transform
    crs = src.crs

    width = src.width
    height = src.height

    nodata = src.nodata

print(f"CRS        : {crs}")
print(f"Resolution : {src.res}")
print(f"Size       : {width} x {height}")

# ============================================================
# READ + ALIGN GHI
# ============================================================

print("\nReading GHI raster...")

with rasterio.open(GHI_RASTER) as ghi_src:

    print(f"GHI CRS        : {ghi_src.crs}")
    print(f"GHI Resolution : {ghi_src.res}")

    ghi = np.full(
        (height, width),
        np.nan,
        dtype=np.float32,
    )

    reproject(
        source=rasterio.band(ghi_src, 1),

        destination=ghi,

        src_transform=ghi_src.transform,
        src_crs=ghi_src.crs,

        dst_transform=transform,
        dst_crs=crs,

        dst_width=width,
        dst_height=height,

        src_nodata=ghi_src.nodata,
        dst_nodata=np.nan,

        resampling=Resampling.bilinear,
    )

# ============================================================
# FILTER GHI
# ============================================================

print("\nApplying GHI threshold...")

# Existing valid pixels
land_mask = landcover != nodata

# GHI >= 4.5
ghi_mask = (
    (~np.isnan(ghi))
    &
    (ghi >= GHI_THRESHOLD)
)

final_mask = (
    land_mask
    &
    ghi_mask
)

# ============================================================
# CREATE OUTPUT
# ============================================================

output = np.full(
    landcover.shape,
    nodata,
    dtype=landcover.dtype,
)

output[final_mask] = landcover[final_mask]

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

# ============================================================
# STATISTICS
# ============================================================

grassland_before = np.sum(landcover == 2)
dry_steppe_before = np.sum(landcover == 3)

grassland_after = np.sum(output == 2)
dry_steppe_after = np.sum(output == 3)

print("\n" + "=" * 70)
print("GHI FILTER SUMMARY")
print("=" * 70)

print(f"Input candidate pixels      : {land_mask.sum():,}")
print(f"GHI >= {GHI_THRESHOLD} pixels     : {ghi_mask.sum():,}")
print(f"Final candidate pixels      : {final_mask.sum():,}")

print()

print(f"Grassland before filtering  : {grassland_before:,}")
print(f"Grassland after filtering   : {grassland_after:,}")

print()

print(f"Dry Steppe before filtering : {dry_steppe_before:,}")
print(f"Dry Steppe after filtering  : {dry_steppe_after:,}")

print()

print(f"Total remaining pixels      : {grassland_after + dry_steppe_after:,}")

print("\nOutput written to:")
print(OUTPUT_RASTER)

print("=" * 70)