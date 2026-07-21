"""
calculate_elevation.py - Add elevation statistics to grassland/dry steppe polygons
OPTIMIZED: No raster reprojection, only polygon reprojection
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.mask import mask
from rasterio.crs import CRS
from rasterstats import zonal_stats
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from shapely.geometry import box

# ============================================
# CONFIGURATION SECTION
# ============================================

# Input paths
INPUT_POLYGONS = Path("outputs/vector/grassland_dry_steppe_with_ghi.gpkg")
ELEVATION_RASTER = Path("data/raw/terrain/aoi_elevation_30m.tif")

# Output paths
OUTPUT_GPKG = Path("outputs/vector/grassland_dry_steppe_with_ghi_elevation.gpkg")

# Create output directories
OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

# --- CONFIGURATION ---
BATCH_SIZE = 5000  # Process polygons in smaller batches to manage memory
# --------------------

print("\n" + "="*60)
print("⛰️ ELEVATION + GHI PROCESSING PIPELINE")
print("="*60)

# ============================================
# STEP 1: READ EXISTING POLYGONS
# ============================================

print("\n=== STEP 1: Reading existing polygons ===")
gdf = gpd.read_file(INPUT_POLYGONS)
print(f"  Loaded {len(gdf):,} polygons")
print(f"  Polygon CRS: {gdf.crs}")
print(f"  Polygon bounds: {gdf.total_bounds}")

# ============================================
# STEP 2: READ ELEVATION RASTER
# ============================================

print("\n=== STEP 2: Reading elevation raster ===")
with rasterio.open(ELEVATION_RASTER) as src:
    raster_crs = src.crs
    raster_bounds = src.bounds
    raster_shape = src.shape
    raster_nodata = src.nodata
    
    print(f"  Elevation CRS: {raster_crs}")
    print(f"  Elevation shape: {raster_shape}")
    print(f"  Elevation bounds: {raster_bounds}")
    print(f"  Elevation nodata: {raster_nodata}")
    
    # Read a small sample to understand data
    elev_data = src.read(1)
    valid_data = elev_data[~np.isnan(elev_data)]
    if len(valid_data) > 0:
        print(f"  Elevation value range: {np.nanmin(elev_data):.2f} - {np.nanmax(elev_data):.2f}")
        print(f"  Elevation mean: {np.nanmean(elev_data):.2f}")
        print(f"  Valid pixels: {len(valid_data):,}")

# ============================================
# STEP 3: REPROJECT POLYGONS TO RASTER CRS (NOT THE RASTER!)
# ============================================

print("\n=== STEP 3: Reprojecting polygons to match raster CRS ===")

if gdf.crs != raster_crs:
    print(f"  Reprojecting polygons from {gdf.crs} to {raster_crs}")
    print("  This is much faster than reprojecting the entire raster!")
    
    # Reproject polygons to raster CRS
    gdf_raster_crs = gdf.to_crs(raster_crs)
    
    # Check if polygons overlap with raster
    raster_box = box(raster_bounds.left, raster_bounds.bottom, 
                     raster_bounds.right, raster_bounds.top)
    
    poly_bounds = gdf_raster_crs.total_bounds
    poly_box = box(poly_bounds[0], poly_bounds[1], poly_bounds[2], poly_bounds[3])
    
    if raster_box.intersects(poly_box):
        print(f"  ✅ Polygons overlap with raster extent")
        intersection = raster_box.intersection(poly_box)
        print(f"  Overlap extent: {intersection.bounds}")
    else:
        print(f"  ⚠️ WARNING: Polygons do NOT overlap with raster extent!")
        print(f"  This will result in no elevation data for any polygons.")
        print(f"  Please check that the data covers the same area.")
        
        # Calculate distance between extents
        x_dist = abs(poly_bounds[0] - raster_bounds.left)
        y_dist = abs(poly_bounds[1] - raster_bounds.bottom)
        print(f"  Distance between extents: {x_dist:.2f} x {y_dist:.2f} meters")
        
        # Ask user if they want to continue
        print("\n  Attempting to continue anyway (will result in NaN values)...")
else:
    print("  Polygons already in same CRS as raster")
    gdf_raster_crs = gdf.copy()

# ============================================
# STEP 4: CALCULATE ELEVATION STATISTICS (NO RASTER REPROJECTION!)
# ============================================

print("\n=== STEP 4: Calculating elevation statistics using reprojected polygons ===")

# Process zonal statistics in smaller batches
batch_size = min(BATCH_SIZE, len(gdf_raster_crs))
n_batches = (len(gdf_raster_crs) + batch_size - 1) // batch_size

print(f"  Processing {len(gdf_raster_crs):,} polygons in {n_batches} batches")
print(f"  Using original elevation raster: {ELEVATION_RASTER}")

# Initialize lists for stats
all_stats = []
successful_polygons = 0
failed_polygons = 0

for i in tqdm(range(n_batches), desc="  Processing elevation statistics"):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(gdf_raster_crs))
    batch_gdf = gdf_raster_crs.iloc[start_idx:end_idx]
    
    try:
        # Use zonal_stats with the original raster (not reprojected)
        stats = zonal_stats(
            batch_gdf,
            str(ELEVATION_RASTER),  # Use original raster path
            stats=["mean", "min", "max", "count", "std", "median", "majority"],
            nodata=np.nan,
            all_touched=False,
            raster_invalid_value=np.nan,
        )
        all_stats.extend(stats)
        successful_polygons += len(batch_gdf)
        
    except Exception as e:
        print(f"  ⚠️ Error in batch {i}: {e}")
        
        try:
            # Retry with all_touched=True
            stats = zonal_stats(
                batch_gdf,
                str(ELEVATION_RASTER),
                stats=["mean", "min", "max", "count", "std", "median", "majority"],
                nodata=np.nan,
                all_touched=True,
                raster_invalid_value=np.nan,
            )
            all_stats.extend(stats)
            successful_polygons += len(batch_gdf)
            
        except Exception as e2:
            print(f"  ⚠️ Fallback failed for batch {i}: {e2}")
            # Add placeholder stats with NaN
            placeholder = [{"mean": np.nan, "min": np.nan, "max": np.nan, 
                           "count": 0, "std": np.nan, "median": np.nan, 
                           "majority": np.nan} for _ in range(len(batch_gdf))]
            all_stats.extend(placeholder)
            failed_polygons += len(batch_gdf)

print(f"  Processed {successful_polygons:,} polygons successfully")
if failed_polygons > 0:
    print(f"  ⚠️ {failed_polygons:,} polygons had errors and were set to NaN")

# ============================================
# STEP 5: ADD STATISTICS TO DATAFRAME
# ============================================

print("\n=== STEP 5: Adding elevation statistics to polygons ===")

# Add elevation stats to original GeoDataFrame (maintaining original CRS)
gdf["elevation_mean"] = [s.get("mean", np.nan) if s else np.nan for s in all_stats]
gdf["elevation_min"] = [s.get("min", np.nan) if s else np.nan for s in all_stats]
gdf["elevation_max"] = [s.get("max", np.nan) if s else np.nan for s in all_stats]
gdf["elevation_std"] = [s.get("std", np.nan) if s else np.nan for s in all_stats]
gdf["elevation_median"] = [s.get("median", np.nan) if s else np.nan for s in all_stats]
gdf["elevation_majority"] = [s.get("majority", np.nan) if s else np.nan for s in all_stats]
gdf["elevation_pixel_count"] = [s.get("count", 0) if s else 0 for s in all_stats]

# Check how many polygons have valid elevation data
valid_elevation = gdf["elevation_mean"].notna().sum()
print(f"  {valid_elevation:,} polygons have valid elevation data")
print(f"  {len(gdf) - valid_elevation:,} polygons have NO elevation data")

# ============================================
# STEP 6: CALCULATE ELEVATION CATEGORIES
# ============================================

print("\n=== STEP 6: Adding elevation categories ===")

def get_elevation_category(elevation):
    if pd.isna(elevation):
        return "No Data"
    elif elevation < 1000:
        return "Low (<1000m)"
    elif elevation < 1500:
        return "Medium-Low (1000-1500m)"
    elif elevation < 2000:
        return "Medium (1500-2000m)"
    elif elevation < 2500:
        return "Medium-High (2000-2500m)"
    elif elevation < 3000:
        return "High (2500-3000m)"
    else:
        return "Very High (>3000m)"

gdf["elevation_category"] = gdf["elevation_mean"].apply(get_elevation_category)

# Elevation range
gdf["elevation_range"] = gdf["elevation_max"] - gdf["elevation_min"]

# Elevation z-score
if len(gdf) > 0 and gdf["elevation_mean"].notna().any():
    mean_elev = gdf["elevation_mean"].mean()
    std_elev = gdf["elevation_mean"].std()
    if std_elev > 0:
        gdf["elevation_zscore"] = (gdf["elevation_mean"] - mean_elev) / std_elev

# ============================================
# STEP 7: SAVE FINAL OUTPUT
# ============================================

print("\n=== STEP 7: Saving final output ===")

# Save full version with all statistics
gdf.to_file(OUTPUT_GPKG, driver="GPKG")
print(f"  ✅ Full GeoPackage: {OUTPUT_GPKG}")

# Save lightweight version
light_columns = [
    "class_id", "class_name", "area_ha", 
    "mean_ghi", "elevation_mean", "elevation_category",
    "elevation_min", "elevation_max", "elevation_range"
]
light_columns = [col for col in light_columns if col in gdf.columns]
light_gdf = gdf[light_columns + ["geometry"]].copy()
light_path = OUTPUT_GPKG.parent / "grassland_dry_steppe_ghi_elevation_light.gpkg"
light_gdf.to_file(light_path, driver="GPKG")
print(f"  ✅ Lightweight version: {light_path}")

# Save statistics as CSV
csv_path = OUTPUT_GPKG.parent / "grassland_dry_steppe_elevation_stats.csv"
stats_columns = [
    "class_id", "class_name", "area_ha", "mean_ghi",
    "elevation_mean", "elevation_min", "elevation_max", 
    "elevation_std", "elevation_median", "elevation_range",
    "elevation_pixel_count", "elevation_category"
]
stats_columns = [col for col in stats_columns if col in gdf.columns]
stats_df = gdf[stats_columns].copy()
stats_df.to_csv(csv_path, index=False)
print(f"  ✅ Statistics CSV: {csv_path}")

# Save polygons with no elevation data
no_data_gdf = gdf[gdf["elevation_mean"].isna()]
if len(no_data_gdf) > 0:
    no_data_path = OUTPUT_GPKG.parent / "polygons_no_elevation_data.gpkg"
    no_data_gdf[["class_id", "class_name", "area_ha", "geometry"]].to_file(no_data_path, driver="GPKG")
    print(f"  ℹ️ {len(no_data_gdf):,} polygons without elevation data saved to: {no_data_path}")

# ============================================
# STEP 8: SUMMARY STATISTICS
# ============================================

print("\n" + "="*60)
print("📊 SUMMARY STATISTICS")
print("="*60)

print(f"Total polygons: {len(gdf):,}")
print(f"Polygons with elevation data: {gdf['elevation_mean'].notna().sum():,}")
print(f"Polygons without elevation data: {gdf['elevation_mean'].isna().sum():,}")
print(f"Total area: {gdf['area_ha'].sum():.2f} ha")

print("\nBy class:")
for class_id, class_name in [(2, "Grassland"), (3, "Dry Steppe")]:
    class_data = gdf[gdf['class_id'] == class_id]
    if len(class_data) > 0:
        print(f"\n  {class_name} ({class_id}):")
        print(f"    Polygons: {len(class_data):,}")
        print(f"    Area: {class_data['area_ha'].sum():.2f} ha")
        if 'mean_ghi' in class_data.columns:
            print(f"    Mean GHI: {class_data['mean_ghi'].mean():.2f}")
        
        elev_valid = class_data['elevation_mean'].dropna()
        if len(elev_valid) > 0:
            print(f"    Mean Elevation: {elev_valid.mean():.2f} m")
            print(f"    Elevation Range: {elev_valid.min():.2f} - {elev_valid.max():.2f} m")
        else:
            print(f"    ⚠️ No elevation data for this class")

print("\nElevation Statistics (for polygons with data):")
elev_valid = gdf[gdf['elevation_mean'].notna()]
if len(elev_valid) > 0:
    print(f"  Overall mean: {elev_valid['elevation_mean'].mean():.2f} m")
    print(f"  Overall min: {elev_valid['elevation_min'].min():.2f} m")
    print(f"  Overall max: {elev_valid['elevation_max'].max():.2f} m")
    print(f"  Standard deviation: {elev_valid['elevation_mean'].std():.2f} m")

print("\nElevation Categories:")
cat_counts = gdf['elevation_category'].value_counts()
for cat, count in cat_counts.items():
    if cat != "No Data":
        pct = (count / len(gdf)) * 100
        print(f"  {cat}: {count:,} polygons ({pct:.1f}%)")

if 'mean_ghi' in gdf.columns and 'elevation_mean' in gdf.columns:
    valid_both = gdf[gdf['elevation_mean'].notna() & gdf['mean_ghi'].notna()]
    if len(valid_both) > 0:
        print("\nCombined Statistics (GHI vs Elevation):")
        corr = valid_both['mean_ghi'].corr(valid_both['elevation_mean'])
        print(f"  Correlation between GHI and Elevation: {corr:.3f}")

print("\n" + "="*60)
print("✅ PROCESSING COMPLETE!")
print("="*60)
print(f"\n📂 Output files:")
print(f"   📁 {OUTPUT_GPKG.parent}")
print(f"   ├── grassland_dry_steppe_with_ghi_elevation.gpkg (full results)")
print(f"   ├── grassland_dry_steppe_ghi_elevation_light.gpkg (lightweight)")
print(f"   ├── grassland_dry_steppe_elevation_stats.csv (statistics)")
if len(no_data_gdf) > 0:
    print(f"   └── polygons_no_elevation_data.gpkg (polygons without elevation data)")
print("="*60)

# ============================================
# STEP 9: CREATE VISUALIZATION
# ============================================

try:
    print("\n=== Creating visualizations ===")
    
    # Filter for polygons with valid data
    valid_data = gdf[gdf['elevation_mean'].notna()]
    
    if len(valid_data) > 0 and 'mean_ghi' in valid_data.columns:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Plot 1: GHI vs Elevation scatter
        ax1 = axes[0, 0]
        scatter = ax1.scatter(
            valid_data['elevation_mean'], 
            valid_data['mean_ghi'],
            c=valid_data['area_ha'],
            cmap='viridis',
            s=10,
            alpha=0.6
        )
        ax1.set_xlabel('Mean Elevation (m)')
        ax1.set_ylabel('Mean GHI')
        ax1.set_title('GHI vs Elevation')
        plt.colorbar(scatter, ax=ax1, label='Area (ha)')
        
        # Plot 2: Elevation distribution by class
        ax2 = axes[0, 1]
        if 'class_name' in valid_data.columns:
            valid_data.boxplot(column='elevation_mean', by='class_name', ax=ax2)
            ax2.set_title('Elevation by Land Cover Class')
            ax2.set_xlabel('Land Cover Class')
            ax2.set_ylabel('Elevation (m)')
        
        # Plot 3: Map of elevation categories
        ax3 = axes[1, 0]
        valid_data.plot(column='elevation_category', ax=ax3, legend=True, categorical=True,
                       legend_kwds={'title': 'Elevation Category'})
        ax3.set_title('Elevation Categories by Polygon')
        ax3.set_aspect('equal')
        
        # Plot 4: Elevation map
        ax4 = axes[1, 1]
        valid_data.plot(column='elevation_mean', ax=ax4, legend=True,
                       cmap='terrain', legend_kwds={'label': 'Mean Elevation (m)', 'shrink': 0.8})
        ax4.set_title('Mean Elevation by Polygon')
        ax4.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = OUTPUT_GPKG.parent / "grassland_dry_steppe_elevation_visualization.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Visualization saved: {fig_path}")
    else:
        print("  ⚠️ No valid elevation data for visualization")
    
except ImportError:
    print("  ⚠️ Matplotlib not available, skipping visualization")
except Exception as e:
    print(f"  ⚠️ Could not create visualization: {e}")

print("\n" + "="*60)