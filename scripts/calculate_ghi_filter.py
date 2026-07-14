"""
calculate_ghi_filter.py

Workflow
--------
1. Read slope-filtered land cover raster.
2. Reproject GHI raster to match land cover.
3. Save aligned GHI raster.
4. Polygonize Grassland (2) and Dry Steppe (3).
5. Calculate mean GHI for each polygon.
6. Filter polygons where mean GHI >= 4.5.
7. Save GeoPackage and Shapefile.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling
from rasterstats import zonal_stats
from shapely.geometry import shape

INPUT_RASTER = Path("outputs/rasters/grassland_dry_steppe_slope_lt8.tif")
GHI_RASTER = Path("data/raw/GHI/GHI.tif")

ALIGNED_GHI = Path("outputs/rasters/GHI_32648.tif")

OUTPUT_GPKG = Path("outputs/vector/solar_candidates.gpkg")
OUTPUT_SHP = Path("outputs/vector/solar_candidates.shp")

GHI_THRESHOLD = 4.5

OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
ALIGNED_GHI.parent.mkdir(parents=True, exist_ok=True)

print("\\nReading land-cover raster...")

with rasterio.open(INPUT_RASTER) as src:
    lc = src.read(1)
    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs
    width = src.width
    height = src.height
    nodata = src.nodata

print("Reprojecting GHI...")

with rasterio.open(GHI_RASTER) as src:
    ghi = np.full((height, width), np.nan, dtype=np.float32)

    reproject(
        source=rasterio.band(src, 1),
        destination=ghi,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=crs,
        dst_width=width,
        dst_height=height,
        src_nodata=src.nodata,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

profile.update(dtype="float32", nodata=np.nan, compress="lzw")

with rasterio.open(ALIGNED_GHI, "w", **profile) as dst:
    dst.write(ghi, 1)

print("Aligned GHI saved:", ALIGNED_GHI)

print("Polygonizing...")

mask = np.isin(lc, [2, 3])

records = []
for geom, value in shapes(lc, mask=mask, transform=transform):
    records.append(
        {
            "geometry": shape(geom),
            "class_id": int(value),
        }
    )

gdf = gpd.GeoDataFrame(records, crs=crs)

print(f"Created {len(gdf):,} polygons")

gdf["class_name"] = gdf["class_id"].map(
    {
        2: "Grassland",
        3: "Dry Steppe",
    }
)

gdf["area_m2"] = gdf.area
gdf["area_ha"] = gdf.area / 10000
gdf["area_km2"] = gdf.area / 1e6

print("Calculating Mean GHI...")

stats = zonal_stats(
    gdf,
    ALIGNED_GHI,
    stats=["mean"],
    nodata=np.nan,
)

gdf["mean_ghi"] = [s["mean"] for s in stats]

before = len(gdf)

gdf = gdf[gdf["mean_ghi"] >= GHI_THRESHOLD].copy()

after = len(gdf)

print(f"Remaining polygons: {after:,}")

gdf.to_file(OUTPUT_GPKG, driver="GPKG")
gdf.to_file(OUTPUT_SHP)

print("\\nDone")
print("=" * 60)
print(f"Input polygons : {before:,}")
print(f"Output polygons: {after:,}")
print(f"GHI Threshold  : {GHI_THRESHOLD}")
print("Saved:")
print(" ", OUTPUT_GPKG)
print(" ", OUTPUT_SHP)
