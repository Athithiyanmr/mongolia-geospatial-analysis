"""
high_potential.py - Extract MPZ class areas and percentages for each polygon
CPU version with fixed reprojection handling.

Input: outputs/vector/technical_potential/technical_potential_polygons.gpkg
MPZ raster: data/raw/LULC/MPZ_v2.tif (reprojected to EPSG:32648 if needed)
Output: outputs/vector/technical_potential_with_mpz/
"""

import os
# CRITICAL: Disable rasterio's disk space check BEFORE importing rasterio
os.environ['CHECK_DISK_FREE_SPACE'] = 'FALSE'

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
import rasterio
from rasterio import features
from rasterio.warp import calculate_default_transform, reproject, Resampling
import time
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================

INPUT_GPKG = Path("outputs/vector/technical_potential/technical_potential_polygons.gpkg")
MPZ_RASTER_ORIG = Path("data/raw/LULC/MPZ_v2.tif")
MPZ_RASTER_REPROJ = Path("outputs/rasters/MPZ_v2_EPSG32648.tif")
TARGET_CRS = "EPSG:32648"

OUTPUT_DIR = Path("outputs/vector/technical_potential_with_mpz/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_GPKG = OUTPUT_DIR / "technical_potential_with_mpz.gpkg"
OUTPUT_STATS = OUTPUT_DIR / "mpz_statistics.csv"

# ============================================
# STEP 0: Determine which raster to use (skip reproj if already in target CRS)
# ============================================

def get_raster_crs(raster_path):
    with rasterio.open(raster_path) as src:
        return src.crs

# Check original raster CRS
orig_crs = get_raster_crs(MPZ_RASTER_ORIG)
if orig_crs == TARGET_CRS:
    MPZ_RASTER = MPZ_RASTER_ORIG
    print(f"  ✅ Original MPZ raster already in {TARGET_CRS}. Using original.")
else:
    # Check if reprojected version already exists and has correct CRS
    if MPZ_RASTER_REPROJ.exists():
        reproj_crs = get_raster_crs(MPZ_RASTER_REPROJ)
        if reproj_crs == TARGET_CRS:
            MPZ_RASTER = MPZ_RASTER_REPROJ
            print(f"  ✅ Reprojected MPZ raster already exists: {MPZ_RASTER_REPROJ}")
        else:
            # Delete and re-reproject
            MPZ_RASTER_REPROJ.unlink()
            MPZ_RASTER = None
    else:
        MPZ_RASTER = None

    # Reproject if needed
    if MPZ_RASTER is None:
        print(f"  Reprojecting {MPZ_RASTER_ORIG} to {TARGET_CRS}...")
        with rasterio.open(MPZ_RASTER_ORIG) as src:
            transform, width, height = calculate_default_transform(
                src.crs, TARGET_CRS, src.width, src.height,
                *src.bounds, resolution=src.res[0]
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': TARGET_CRS,
                'transform': transform,
                'width': width,
                'height': height,
                'res': (abs(transform.a), abs(transform.e))
            })
            MPZ_RASTER_REPROJ.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(MPZ_RASTER_REPROJ, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=TARGET_CRS,
                    resampling=Resampling.nearest
                )
        MPZ_RASTER = MPZ_RASTER_REPROJ
        print(f"  ✅ Reprojected raster saved to: {MPZ_RASTER}")

# ============================================
# FUNCTIONS
# ============================================

def get_raster_info(raster_path):
    with rasterio.open(raster_path) as src:
        return {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'nodata': src.nodata,
            'bounds': src.bounds,
            'res': src.res
        }

def extract_polygon_mpz_stats(polygon, raster_path, raster_info, class_values):
    """Extract MPZ stats using CPU."""
    geom = polygon.geometry
    with rasterio.open(raster_path) as src:
        mask = features.geometry_mask(
            [geom],
            out_shape=(src.height, src.width),
            transform=src.transform,
            all_touched=True,
            invert=False
        )
        data = src.read(1)
        masked_data = data[~mask]  # pixels inside polygon
        pixel_area_km2 = abs(src.res[0] * src.res[1]) / 1_000_000

    unique, counts = np.unique(masked_data, return_counts=True)
    total_pixels = counts.sum()
    results = {}
    for val in class_values:
        idx = np.where(unique == val)[0]
        if len(idx) > 0:
            count = counts[idx[0]]
            area_km2 = count * pixel_area_km2
            pct = (count / total_pixels) * 100 if total_pixels > 0 else 0
        else:
            area_km2 = 0.0
            pct = 0.0
        results[f'mpz_{val}_km2'] = area_km2
        results[f'mpz_{val}_pct'] = pct
    results['polygon_area_km2'] = polygon.geometry.area / 1_000_000
    return results

# ============================================
# MAIN
# ============================================

print("="*60)
print("📊 EXTRACTING MPZ CLASS STATISTICS (CPU)")
print("="*60)
print(f"\n📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Check inputs
if not INPUT_GPKG.exists():
    print(f"  ❌ Input file not found: {INPUT_GPKG}")
    exit(1)
if not MPZ_RASTER.exists():
    print(f"  ❌ MPZ raster not found: {MPZ_RASTER}")
    exit(1)

# STEP 1: Read polygons
print("\n=== STEP 1: Reading technical potential polygons ===")
gdf = gpd.read_file(INPUT_GPKG)
print(f"  Loaded {len(gdf):,} polygons")
print(f"  CRS: {gdf.crs}")

# STEP 2: Read MPZ raster info
print("\n=== STEP 2: Reading MPZ raster ===")
raster_info = get_raster_info(MPZ_RASTER)
print(f"  CRS: {raster_info['crs']}")
print(f"  Resolution: {raster_info['res']}")
print(f"  Shape: {raster_info['width']} x {raster_info['height']}")

with rasterio.open(MPZ_RASTER) as src:
    data = src.read(1)
    unique_values = np.unique(data)
    if raster_info['nodata'] is not None:
        unique_values = unique_values[unique_values != raster_info['nodata']]
    class_values = sorted(unique_values)
    print(f"  Unique MPZ classes found: {class_values}")

# STEP 3: Align CRS
print("\n=== STEP 3: CRS alignment ===")
if gdf.crs != raster_info['crs']:
    print(f"  Reprojecting polygons to raster CRS ({raster_info['crs']})...")
    gdf = gdf.to_crs(raster_info['crs'])
else:
    print("  ✅ CRS matched")

# STEP 4: Extract MPZ stats
print("\n=== STEP 4: Extracting MPZ class areas and percentages ===")
print(f"  Processing {len(gdf):,} polygons...")
results_list = []
for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="  Processing polygons"):
    stats = extract_polygon_mpz_stats(row, MPZ_RASTER, raster_info, class_values)
    results_list.append(stats)

mpz_df = pd.DataFrame(results_list)
for col in mpz_df.columns:
    gdf[col] = mpz_df[col].values

# STEP 5: Save output
print("\n=== STEP 5: Saving output ===")
gdf.to_file(OUTPUT_GPKG, driver="GPKG")
print(f"  ✅ Saved: {OUTPUT_GPKG}")
print(f"     Polygons: {len(gdf):,}")
print(f"     Columns added: {len(mpz_df.columns)}")

# Statistics summary
stats_summary = []
for val in class_values:
    km2_col = f'mpz_{val}_km2'
    pct_col = f'mpz_{val}_pct'
    if km2_col in gdf.columns:
        total_km2 = gdf[km2_col].sum()
        mean_pct = gdf[pct_col].mean()
        stats_summary.append({
            'mpz_class': val,
            'total_area_km2': total_km2,
            'mean_percentage': mean_pct,
            'polygons_with_class': (gdf[km2_col] > 0).sum()
        })
summary_df = pd.DataFrame(stats_summary)
summary_df.to_csv(OUTPUT_STATS, index=False)
print(f"  ✅ Statistics saved: {OUTPUT_STATS}")

# STEP 6: Summary
print("\n" + "="*60)
print("📊 MPZ STATISTICS SUMMARY")
print("="*60)
print(f"\nTotal technical potential polygons: {len(gdf):,}")
print(f"Total area (from GIS): {gdf.geometry.area.sum() / 1_000_000:.2f} km²")
print(f"Total area (sum of MPZ classes): {gdf[[f'mpz_{v}_km2' for v in class_values]].sum().sum():.2f} km²")
print("\nClass-wise totals:")
for val in class_values:
    km2_col = f'mpz_{val}_km2'
    if km2_col in gdf.columns:
        total_km2 = gdf[km2_col].sum()
        pct_of_total = (total_km2 / gdf.geometry.area.sum() * 1_000_000) * 100
        print(f"  MPZ {val}: {total_km2:.2f} km² ({pct_of_total:.1f}% of total polygon area)")

print("\n✅ MPZ extraction complete!")
print(f"📁 Output folder: {OUTPUT_DIR}")
print(f"   ├── {OUTPUT_GPKG.name}")
print(f"   └── {OUTPUT_STATS.name}")
print("="*60)