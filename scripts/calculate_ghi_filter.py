"""
calculate_ghi_filter.py - Optimized for large rasters with BIGTIFF support
MODIFIED: Keeps ALL polygons even without GHI data (sets to NaN)
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterstats import zonal_stats
from shapely.geometry import shape
from shapely.ops import transform
import pyproj
import pandas as pd
from tqdm import tqdm
import os

# ============================================
# CONFIGURATION SECTION
# ============================================

# Input paths
INPUT_RASTER = Path("outputs/rasters/grassland_dry_steppe_slope_lt8_filled.tif")
GHI_RASTER = Path("data/raw/GHI/GHI.tif")

# Output paths
ALIGNED_GHI = Path("outputs/rasters/GHI_32648.tif")
LANDCOVER_POLYGONS = Path("outputs/vector/grassland_dry_steppe_polygons.gpkg")
OUTPUT_GPKG = Path("outputs/vector/grassland_dry_steppe_with_ghi.gpkg")

# Create output directories
for path in [ALIGNED_GHI.parent, LANDCOVER_POLYGONS.parent, OUTPUT_GPKG.parent]:
    path.mkdir(parents=True, exist_ok=True)

# --- CONFIGURATION ---
MIN_AREA_HA = 0.1  # Minimum polygon area in hectares
BATCH_SIZE = 50000  # Process polygons in batches to manage memory
CLASS_MAPPING = {2: "Grassland", 3: "Dry Steppe"}  # Land cover class mapping
KEEP_ALL_POLYGONS = True  # ✅ NEW: Keep all polygons even without GHI data
# --------------------

print("\n" + "="*60)
print("🌾 LAND COVER + GHI PROCESSING PIPELINE")
print("="*60)
print(f"  📌 Keeping ALL polygons (including those without GHI data)")

# ============================================
# STEP 1: READ LAND COVER RASTER
# ============================================

print("\n=== STEP 1: Reading land-cover raster ===")
with rasterio.open(INPUT_RASTER) as src:
    lc = src.read(1)
    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs
    width = src.width
    height = src.height
    
    # Get pixel resolution to calculate minimum area
    pixel_area_m2 = abs(transform[0] * transform[4])  # pixel size in meters
    min_area_m2 = MIN_AREA_HA * 10000  # Convert ha to m²
    min_pixels = int(min_area_m2 / pixel_area_m2) + 1
    
    print(f"  Raster size: {width:,} x {height:,} pixels")
    print(f"  Pixel resolution: {abs(transform[0]):.2f} m")
    print(f"  Pixel area: {pixel_area_m2:.2f} m²")
    print(f"  Minimum area: {MIN_AREA_HA} ha ({min_area_m2:.0f} m²)")
    print(f"  Minimum pixels: {min_pixels}")

# ============================================
# STEP 2: REPROJECT GHI RASTER WITH BIGTIFF
# ============================================

print("\n=== STEP 2: Reprojecting GHI with BIGTIFF support ===")

# First, check if aligned GHI already exists
if ALIGNED_GHI.exists():
    print(f"  Aligned GHI already exists: {ALIGNED_GHI}")
    print("  Skipping reprojection. Delete the file to reprocess.")
else:
    with rasterio.open(GHI_RASTER) as src:
        print(f"  GHI original CRS: {src.crs}")
        print(f"  GHI original shape: {src.shape}")
        
        # Create output array
        ghi_reprojected = np.full((height, width), np.nan, dtype=np.float32)
        
        print(f"  Reprojecting to shape: {height} x {width}")
        print("  This may take a while...")
        
        # Reproject using nearest neighbor for better performance with BIGTIFF
        reproject(
            source=rasterio.band(src, 1),
            destination=ghi_reprojected,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            dst_width=width,
            dst_height=height,
            src_nodata=src.nodata if src.nodata is not None else -9999,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        
        print(f"  Reprojected GHI shape: {ghi_reprojected.shape}")
        
        # Save aligned GHI with BIGTIFF support
        print("  Saving aligned GHI with BIGTIFF support...")
        
        # Update profile with BIGTIFF settings
        profile.update({
            "dtype": "float32",
            "nodata": np.nan,
            "compress": "lzw",
            "count": 1,
            "BIGTIFF": "YES",  # ✅ CRITICAL: Enable BIGTIFF
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "predictor": 2,  # Floating point predictor
        })
        
        with rasterio.open(ALIGNED_GHI, "w", **profile) as dst:
            dst.write(ghi_reprojected, 1)
        
        print(f"  ✅ Aligned GHI saved: {ALIGNED_GHI}")
        
        # Get file size
        file_size = ALIGNED_GHI.stat().st_size / (1024**3)
        print(f"  File size: {file_size:.2f} GB")

# ============================================
# STEP 3: POLYGONIZE LAND COVER
# ============================================

print("\n=== STEP 3: Polygonizing land cover ===")
mask = np.isin(lc, [2, 3])
print(f"  Pixels with classes 2/3: {np.sum(mask):,}")

# Check if there are any pixels to process
if np.sum(mask) == 0:
    print("❌ No pixels found with classes 2 or 3!")
    print("   Please check the raster values and CLASS_MAPPING configuration.")
    print(f"   Unique values in raster: {np.unique(lc)[:20]}")
    exit()

# Polygonize with connectivity=8 for better polygons
records = []
polygon_count = 0
skipped_count = 0

print("  Polygonizing (this may take a while)...")
for geom, value in tqdm(shapes(lc, mask=mask, transform=transform, connectivity=8), 
                        desc="  Creating polygons"):
    poly = shape(geom)
    
    # Filter out tiny polygons immediately
    if poly.area < min_area_m2:
        skipped_count += 1
        continue
    
    records.append({
        "geometry": poly,
        "class_id": int(value),
    })
    polygon_count += 1

print(f"  Created {polygon_count:,} polygons (skipped {skipped_count:,} tiny polygons)")

if polygon_count == 0:
    print("❌ No polygons created! Check MIN_AREA_HA setting.")
    print(f"   Try reducing MIN_AREA_HA (currently {MIN_AREA_HA} ha)")
    exit()

# ============================================
# STEP 4: CREATE AND SAVE LAND COVER POLYGONS
# ============================================

print("\n=== STEP 4: Creating land cover GeoDataFrame ===")
gdf = gpd.GeoDataFrame(records, crs=crs)
original_count = len(gdf)
print(f"  GeoDataFrame created with {original_count:,} features")

# Add class names
gdf["class_name"] = gdf["class_id"].map(CLASS_MAPPING)

# Save the polygonized land cover
print("  Saving land cover polygons...")
gdf.to_file(LANDCOVER_POLYGONS, driver="GPKG")
print(f"  ✅ Land cover polygons saved: {LANDCOVER_POLYGONS}")

# Also save as shapefile for compatibility (if not too large)
if len(gdf) < 100000:
    shp_path = LANDCOVER_POLYGONS.parent / "grassland_dry_steppe_polygons.shp"
    try:
        gdf.to_file(shp_path)
        print(f"  ✅ Also saved as shapefile: {shp_path}")
    except Exception as e:
        print(f"  ⚠️ Could not save shapefile: {e}")

# ============================================
# STEP 5: CALCULATE AREAS
# ============================================

print("\n=== STEP 5: Calculating areas ===")
gdf["area_m2"] = gdf.area
gdf["area_ha"] = gdf.area / 10000

print(f"  Total area: {gdf['area_ha'].sum():.2f} ha")
print(f"  Polygons > 1 ha: {len(gdf[gdf['area_ha'] >= 1]):,}")
print("\n  Area by class:")
for class_id, class_name in CLASS_MAPPING.items():
    class_area = gdf[gdf['class_id'] == class_id]['area_ha'].sum()
    class_count = len(gdf[gdf['class_id'] == class_id])
    print(f"    {class_name}: {class_area:.2f} ha ({class_count:,} polygons)")

# ============================================
# STEP 6: CALCULATE GHI STATISTICS
# ============================================

print("\n=== STEP 6: Calculating GHI statistics in batches ===")
print(f"  🔄 KEEPING ALL {original_count:,} polygons (setting missing GHI to NaN)")

# Process zonal statistics in batches to avoid memory issues
batch_size = min(10000, len(gdf))  # Use smaller batch if needed
n_batches = (len(gdf) + batch_size - 1) // batch_size

print(f"  Processing {len(gdf):,} polygons in {n_batches} batches")

all_stats = []
failed_polygons = 0

for i in tqdm(range(n_batches), desc="  Processing GHI statistics"):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(gdf))
    batch_gdf = gdf.iloc[start_idx:end_idx]
    
    try:
        stats = zonal_stats(
            batch_gdf,
            str(ALIGNED_GHI),
            stats=["mean", "min", "max", "count"],
            nodata=np.nan,
            all_touched=False,
        )
        all_stats.extend(stats)
    except Exception as e:
        print(f"  ⚠️ Error in batch {i}: {e}")
        # If raster too large, try with all_touched=True
        try:
            stats = zonal_stats(
                batch_gdf,
                str(ALIGNED_GHI),
                stats=["mean", "min", "max", "count"],
                nodata=np.nan,
                all_touched=True,
            )
            all_stats.extend(stats)
        except Exception as e2:
            print(f"  ⚠️ Fallback failed for batch {i}: {e2}")
            # Add placeholder stats with NaN
            placeholder = [{"mean": np.nan, "min": np.nan, "max": np.nan, "count": 0} 
                          for _ in range(len(batch_gdf))]
            all_stats.extend(placeholder)
            failed_polygons += len(batch_gdf)

print(f"  Calculated stats for {len(all_stats):,} polygons")

# Add stats to GeoDataFrame
gdf["mean_ghi"] = [s.get("mean", np.nan) if s else np.nan for s in all_stats]
gdf["min_ghi"] = [s.get("min", np.nan) if s else np.nan for s in all_stats]
gdf["max_ghi"] = [s.get("max", np.nan) if s else np.nan for s in all_stats]
gdf["pixel_count"] = [s.get("count", 0) if s else 0 for s in all_stats]

# ✅ MODIFIED: DO NOT drop polygons with no GHI data
# Instead, keep them with NaN values
before = len(gdf)
# gdf = gdf.dropna(subset=["mean_ghi"])  # ❌ REMOVED THIS LINE
after = len(gdf)

print(f"  ✅ All {after:,} polygons preserved")
if failed_polygons > 0:
    print(f"  ⚠️ {failed_polygons:,} polygons had no GHI data (set to NaN)")

# Count polygons with and without GHI data
ghi_valid = gdf["mean_ghi"].notna().sum()
ghi_missing = len(gdf) - ghi_valid
print(f"  Polygons with GHI data: {ghi_valid:,} ({ghi_valid/len(gdf)*100:.1f}%)")
print(f"  Polygons WITHOUT GHI data: {ghi_missing:,} ({ghi_missing/len(gdf)*100:.1f}%)")

# ============================================
# STEP 7: SAVE FINAL OUTPUT
# ============================================

print("\n=== STEP 7: Saving final output ===")

# Save full version to GeoPackage
gdf.to_file(OUTPUT_GPKG, driver="GPKG")
print(f"  ✅ Full GeoPackage: {OUTPUT_GPKG}")

# Save lightweight version
light_gdf = gdf[["class_id", "class_name", "area_ha", "mean_ghi", "geometry"]]
light_path = OUTPUT_GPKG.parent / "grassland_dry_steppe_ghi_light.gpkg"
light_gdf.to_file(light_path, driver="GPKG")
print(f"  ✅ Lightweight version: {light_path}")

# Save as shapefile (if not too large)
if len(gdf) < 100000:
    shp_output = OUTPUT_GPKG.parent / "grassland_dry_steppe_with_ghi.shp"
    try:
        gdf.to_file(shp_output)
        print(f"  ✅ Shapefile version: {shp_output}")
    except Exception as e:
        print(f"  ⚠️ Could not save shapefile: {e}")

# Save statistics as CSV
csv_path = OUTPUT_GPKG.parent / "grassland_dry_steppe_ghi_stats.csv"
stats_df = gdf[["class_id", "class_name", "area_ha", "mean_ghi", "min_ghi", "max_ghi", "pixel_count"]]
stats_df.to_csv(csv_path, index=False)
print(f"  ✅ Statistics CSV: {csv_path}")

# Save polygons with missing GHI data separately
if ghi_missing > 0:
    missing_ghi_gdf = gdf[gdf["mean_ghi"].isna()]
    missing_path = OUTPUT_GPKG.parent / "polygons_missing_ghi_data.gpkg"
    missing_ghi_gdf[["class_id", "class_name", "area_ha", "geometry"]].to_file(missing_path, driver="GPKG")
    print(f"  ℹ️ {ghi_missing:,} polygons without GHI data saved to: {missing_path}")

# ============================================
# STEP 8: SUMMARY STATISTICS
# ============================================

print("\n" + "="*60)
print("📊 SUMMARY STATISTICS")
print("="*60)

print(f"Total polygons: {len(gdf):,}")
print(f"Total area: {gdf['area_ha'].sum():.2f} ha")
print(f"Polygons with GHI data: {ghi_valid:,} ({ghi_valid/len(gdf)*100:.1f}%)")
print(f"Polygons WITHOUT GHI data: {ghi_missing:,} ({ghi_missing/len(gdf)*100:.1f}%)")

print("\nBy class:")
for class_id, class_name in CLASS_MAPPING.items():
    class_data = gdf[gdf['class_id'] == class_id]
    if len(class_data) > 0:
        print(f"\n  {class_name}:")
        print(f"    Polygons: {len(class_data):,}")
        print(f"    Area: {class_data['area_ha'].sum():.2f} ha")
        
        # GHI statistics only for polygons with data
        ghi_data = class_data[class_data["mean_ghi"].notna()]
        if len(ghi_data) > 0:
            print(f"    Polygons with GHI: {len(ghi_data):,}")
            print(f"    Mean GHI: {ghi_data['mean_ghi'].mean():.2f}")
            print(f"    GHI Range: {ghi_data['mean_ghi'].min():.2f} - {ghi_data['mean_ghi'].max():.2f}")
        else:
            print(f"    ⚠️ No GHI data for this class")

print("\nGHI Statistics (for polygons with data):")
ghi_data = gdf[gdf["mean_ghi"].notna()]
if len(ghi_data) > 0:
    print(f"  Overall mean: {ghi_data['mean_ghi'].mean():.2f}")
    print(f"  Overall min: {ghi_data['mean_ghi'].min():.2f}")
    print(f"  Overall max: {ghi_data['mean_ghi'].max():.2f}")
    print(f"  Standard deviation: {ghi_data['mean_ghi'].std():.2f}")

print("\n" + "="*60)
print("✅ PROCESSING COMPLETE!")
print("="*60)
print(f"\n📂 Output files:")
print(f"   📁 {OUTPUT_GPKG.parent}")
print(f"   ├── grassland_dry_steppe_polygons.gpkg (land cover polygons - ALL {len(gdf):,})")
print(f"   ├── grassland_dry_steppe_with_ghi.gpkg (final results - ALL {len(gdf):,})")
print(f"   ├── grassland_dry_steppe_ghi_light.gpkg (lightweight version - ALL {len(gdf):,})")
print(f"   ├── grassland_dry_steppe_ghi_stats.csv (statistics)")
print(f"   ├── GHI_32648.tif (aligned GHI raster - BIGTIFF)")
if ghi_missing > 0:
    print(f"   └── polygons_missing_ghi_data.gpkg ({ghi_missing:,} polygons without GHI)")
print("="*60)

# ============================================
# STEP 9: QUICK VISUALIZATION (Optional)
# ============================================

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    print("\n=== Creating quick visualization ===")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Land cover classes
    ax1 = axes[0]
    gdf.plot(column='class_id', ax=ax1, legend=True, categorical=True,
             cmap='Set2', legend_kwds={'labels': list(CLASS_MAPPING.values())})
    ax1.set_title(f'Land Cover Classes\n({len(gdf):,} polygons)')
    ax1.set_aspect('equal')
    
    # Plot 2: Mean GHI (only polygons with data)
    ax2 = axes[1]
    ghi_data = gdf[gdf["mean_ghi"].notna()]
    if len(ghi_data) > 0:
        ghi_data.plot(column='mean_ghi', ax=ax2, legend=True,
                     cmap='RdYlBu_r', legend_kwds={'label': 'Mean GHI', 'shrink': 0.8})
        ax2.set_title(f'Mean GHI by Polygon\n({len(ghi_data):,} polygons with data)')
    else:
        ax2.text(0.5, 0.5, 'No GHI data available', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Mean GHI by Polygon (No Data)')
    ax2.set_aspect('equal')
    
    # Plot 3: Data completeness (which polygons have GHI data)
    ax3 = axes[2]
    gdf['has_ghi'] = gdf['mean_ghi'].notna()
    gdf.plot(column='has_ghi', ax=ax3, legend=True, categorical=True,
             cmap='RdYlGn', legend_kwds={'labels': ['No GHI', 'Has GHI']})
    ax3.set_title(f'GHI Data Completeness\n({ghi_valid:,} with GHI, {ghi_missing:,} without)')
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = OUTPUT_GPKG.parent / "grassland_dry_steppe_ghi_visualization.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Visualization saved: {fig_path}")
    
except ImportError:
    print("  ⚠️ Matplotlib not available, skipping visualization")
except Exception as e:
    print(f"  ⚠️ Could not create visualization: {e}")

print("\n" + "="*60)