from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterstats import zonal_stats
from shapely.geometry import shape

# ============================================================
# INPUTS
# ============================================================

INPUT_RASTER = Path(
    "outputs/rasters/grassland_dry_steppe_slope_lt8.tif"
)

GHI_RASTER = Path(
    "data/raw/GHI/GHI.tif"
)

OUTPUT_GPKG = Path(
    "outputs/vector/solar_candidates.gpkg"
)

OUTPUT_SHP = Path(
    "outputs/vector/solar_candidates.shp"
)

GHI_THRESHOLD = 4.5

# ============================================================
# READ LAND COVER
# ============================================================

print("\nReading raster...")

with rasterio.open(INPUT_RASTER) as src:

    raster = src.read(1)
    transform = src.transform
    crs = src.crs
    nodata = src.nodata

# ============================================================
# POLYGONIZE
# ============================================================

print("Polygonizing raster...")

mask = np.isin(raster, [2, 3])

features = []

for geom, value in shapes(
    raster,
    mask=mask,
    transform=transform,
):

    features.append(
        {
            "geometry": shape(geom),
            "class": int(value),
        }
    )

gdf = gpd.GeoDataFrame(
    features,
    crs=crs,
)

print(f"Polygons created : {len(gdf):,}")

# ============================================================
# AREA
# ============================================================

gdf["area_m2"] = gdf.area
gdf["area_ha"] = gdf.area / 10000
gdf["area_km2"] = gdf.area / 1e6

# ============================================================
# MEAN GHI
# ============================================================

print("Calculating Mean GHI...")

stats = zonal_stats(
    gdf,
    GHI_RASTER,
    stats=["mean"],
    nodata=None,
)

gdf["mean_ghi"] = [
    s["mean"] for s in stats
]

# ============================================================
# FILTER
# ============================================================

before = len(gdf)

gdf = gdf[
    gdf["mean_ghi"] >= GHI_THRESHOLD
].copy()

after = len(gdf)

print(f"Remaining polygons : {after:,}")

# ============================================================
# CLASS NAMES
# ============================================================

gdf["class_name"] = gdf["class"].map(
    {
        2: "Grassland",
        3: "Dry Steppe",
    }
)

# ============================================================
# SAVE
# ============================================================

OUTPUT_GPKG.parent.mkdir(
    parents=True,
    exist_ok=True,
)

gdf.to_file(
    OUTPUT_GPKG,
    driver="GPKG",
)

gdf.to_file(
    OUTPUT_SHP,
)

print("\nDone")
print("=" * 60)
print(f"Input polygons : {before:,}")
print(f"Output polygons: {after:,}")
print(f"Saved : {OUTPUT_GPKG}")
print(f"Saved : {OUTPUT_SHP}")