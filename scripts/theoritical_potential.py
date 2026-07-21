"""
calculate_and_filter_theoretical_potential.py - Complete pipeline
Step 1: Calculate minimum distances to infrastructure (excluding roads)
Step 2: Filter polygons based on distance criteria
Step 3: Output theoretical potential polygons and total area
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime

# ============================================
# GPU DETECTION - PyTorch
# ============================================

GPU_AVAILABLE = False

try:
    import torch
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        print("="*60)
        print("✅ GPU Acceleration Available!")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("="*60)
    else:
        print("ℹ️ Running in CPU mode")
except ImportError:
    print("ℹ️ PyTorch not installed. Running in CPU mode.")

# ============================================
# CONFIGURATION
# ============================================

INPUT_POLYGONS = Path("outputs/vector/grassland_dry_steppe_with_ghi_elevation.gpkg")

# Infrastructure paths (NO ROADS - excluded)
HIGHWAYS = Path("data/raw/aoi_clipped/road/OSM/mn_highway.shp")
POWERLINES = Path("data/raw/aoi_clipped/powerline/powerline_clipped.shp")
PROTECTED_LANDS = Path("data/raw/aoi_clipped/legal protected/legal_protected_sites_clipped.shp")
RAILWAYS = Path("data/raw/aoi_clipped/railway/railway_clipped.shp")
SUBSTATIONS = Path("data/raw/aoi_clipped/substation/substations_clipped.shp")
AIRPORTS = Path("data/raw/aoi_clipped/airport/airport_clipped.shp")

# LULC data path
LULC_FILLED_PATH = Path("outputs/rasters/LULC_2025_AOI_EPSG32648_filled.tif")

# Output paths
DISTANCES_GPKG = Path("outputs/vector/grassland_dry_steppe_with_distances.gpkg")

# Theoretical potential output
THEORETICAL_DIR = Path("outputs/vector/theoretical_potential/")
THEORETICAL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILTERED = THEORETICAL_DIR / "theoretical_potential_polygons.gpkg"
OUTPUT_STATS = THEORETICAL_DIR / "filtering_statistics.csv"
OUTPUT_REMOVED = THEORETICAL_DIR / "removed_polygons.gpkg"
OUTPUT_SUMMARY = THEORETICAL_DIR / "filtering_summary.txt"

# --- CONFIGURATION ---
WATER_CLASS_VALUE = 0  # Water class in LULC
# NO SEARCH BUFFER - finds nearest infrastructure regardless of distance

# Filtering thresholds (in meters)
FILTERS = {
    'min_dist_waterbody': 100,      # > 100m from waterbodies
    'min_dist_railway': 200,        # > 200m from railways
    'min_dist_highway': 500,        # > 500m from highways
    'min_dist_airport': 500,        # > 500m from airports
    'min_dist_protected': 1000,     # > 1km from protected lands
}
# --------------------------------

print("\n" + "="*60)
print("🏗️  COMPLETE PIPELINE: Distance Calculation + Filtering")
print("🚫 EXCLUDING: Roads (run separately)")
print("✅ INCLUDING: Highways, Powerlines, Protected, Railways, Substations, Airports, Waterbodies")
print("🔍 NO SEARCH BUFFER - finds nearest infrastructure regardless of distance")
print("="*60)
start_time = time.time()

# ============================================
# STEP 1: CALCULATE DISTANCES (if not already done)
# ============================================

if DISTANCES_GPKG.exists():
    print(f"\n✅ Distances file already exists: {DISTANCES_GPKG}")
    print("   Loading existing distances...")
    gdf = gpd.read_file(DISTANCES_GPKG)
else:
    print("\n=== STEP 1: Calculating minimum distances (no search buffer) ===")
    
    # Read polygons
    print("\n  Reading polygons...")
    gdf = gpd.read_file(INPUT_POLYGONS)
    print(f"  Loaded {len(gdf):,} polygons")
    
    # Read infrastructure (NO ROADS)
    print("\n  Reading infrastructure (roads excluded)...")
    
    def read_infrastructure(file_path, name):
        try:
            if file_path.exists():
                gdf_infra = gpd.read_file(file_path)
                if len(gdf_infra) > 0:
                    # Convert polygons to centroids
                    if gdf_infra.geometry.iloc[0].geom_type in ['Polygon', 'MultiPolygon']:
                        gdf_infra['geometry'] = gdf_infra.geometry.centroid
                        gdf_infra = gdf_infra.set_geometry('geometry')
                    
                    # Reproject
                    if gdf_infra.crs != gdf.crs:
                        gdf_infra = gdf_infra.to_crs(gdf.crs)
                    
                    print(f"    ✅ {name}: {len(gdf_infra):,}")
                    return gdf_infra
                return None
            else:
                print(f"    ⚠️ {name} not found")
                return None
        except Exception as e:
            print(f"    ❌ Error loading {name}: {e}")
            return None
    
    infrastructure = {}
    infrastructure['highways'] = read_infrastructure(HIGHWAYS, "Highways")
    infrastructure['powerlines'] = read_infrastructure(POWERLINES, "Powerlines")
    infrastructure['protected'] = read_infrastructure(PROTECTED_LANDS, "Protected Lands")
    infrastructure['railways'] = read_infrastructure(RAILWAYS, "Railways")
    infrastructure['substations'] = read_infrastructure(SUBSTATIONS, "Substations")
    infrastructure['airports'] = read_infrastructure(AIRPORTS, "Airports")
    
    # Extract waterbodies from LULC
    print("\n  Extracting waterbodies from LULC...")
    try:
        import rasterio
        from rasterio import features
        from shapely.geometry import shape
        import numpy as np
        
        if LULC_FILLED_PATH.exists():
            with rasterio.open(LULC_FILLED_PATH) as src:
                lulc_data = src.read(1)
                if lulc_data.dtype in ['float64', 'float32']:
                    water_mask = np.isclose(lulc_data, float(WATER_CLASS_VALUE))
                else:
                    water_mask = (lulc_data == WATER_CLASS_VALUE)
                
                water_pixel_count = int(water_mask.sum())
                print(f"    Water pixels: {water_pixel_count:,}")
                
                if water_pixel_count > 0:
                    water_mask_uint8 = water_mask.astype(np.uint8)
                    shapes_gen = features.shapes(water_mask_uint8, mask=water_mask_uint8, transform=src.transform)
                    
                    water_features = []
                    for geom, value in shapes_gen:
                        if value == 1:
                            try:
                                polygon = shape(geom)
                                if polygon.is_valid and not polygon.is_empty:
                                    water_features.append({'geometry': polygon})
                            except:
                                continue
                    
                    if water_features:
                        water_gdf = gpd.GeoDataFrame(water_features, crs=src.crs)
                        if water_gdf.crs != gdf.crs:
                            water_gdf = water_gdf.to_crs(gdf.crs)
                        water_gdf['geometry'] = water_gdf.geometry.centroid
                        water_gdf = water_gdf.set_geometry('geometry')
                        infrastructure['waterbodies'] = water_gdf
                        print(f"    ✅ Waterbodies: {len(water_gdf):,}")
        else:
            print(f"    ⚠️ LULC file not found: {LULC_FILLED_PATH}")
    except Exception as e:
        print(f"    ⚠️ Error extracting waterbodies: {e}")
    
    # Define distance columns (NO ROADS)
    distance_columns = {
        'min_dist_highway': 'highways',
        'min_dist_powerline': 'powerlines',
        'min_dist_protected': 'protected',
        'min_dist_railway': 'railways',
        'min_dist_substation': 'substations',
        'min_dist_airport': 'airports',
        'min_dist_waterbody': 'waterbodies'
    }
    
    # Function to calculate distances (GPU or CPU)
    def calculate_distances_gpu(polygons_gdf, infra_gdf, batch_size=50000):
        try:
            print("      🚀 Using GPU...")
            centroids = polygons_gdf.geometry.centroid
            polygon_coords = np.array([[p.x, p.y] for p in centroids], dtype=np.float32)
            infra_coords = np.array([[p.x, p.y] for p in infra_gdf.geometry], dtype=np.float32)
            
            polygon_coords_gpu = torch.tensor(polygon_coords, device='cuda')
            infra_coords_gpu = torch.tensor(infra_coords, device='cuda')
            
            all_distances = []
            n = len(polygon_coords_gpu)
            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)
                batch = polygon_coords_gpu[i:end]
                for j in range(len(batch)):
                    point = batch[j]
                    diff = infra_coords_gpu - point
                    dist = torch.sqrt(torch.sum(diff * diff, dim=1))
                    all_distances.append(torch.min(dist).item())
            
            del polygon_coords_gpu, infra_coords_gpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return np.array(all_distances)
        except Exception as e:
            print(f"      ⚠️ GPU failed: {e}")
            return None
    
    def calculate_distances_cpu(polygons_gdf, infra_gdf):
        print("      💻 Using CPU (no search buffer)...")
        distances = []
        sindex = infra_gdf.sindex
        geometries = infra_gdf.geometry
        centroids = polygons_gdf.geometry.centroid
        
        for idx, row in tqdm(polygons_gdf.iterrows(), total=len(polygons_gdf), desc="        Calculating"):
            polygon = row.geometry
            centroid = centroids.iloc[idx]
            minx, miny, maxx, maxy = polygon.bounds
            possible = list(sindex.intersection((minx, miny, maxx, maxy)))
            if not possible:
                # Fallback: 100km buffer
                buffered = centroid.buffer(100000)
                possible = list(sindex.intersection(buffered.bounds))
            if not possible:
                possible = list(range(len(geometries)))
            
            candidates = geometries.iloc[possible]
            try:
                min_dist = candidates.distance(polygon).min()
                if candidates.intersects(polygon).any():
                    min_dist = 0.0
                distances.append(min_dist)
            except:
                try:
                    min_dist = candidates.distance(centroid).min()
                    distances.append(min_dist)
                except:
                    distances.append(np.nan)
        return np.array(distances)
    
    def calculate_distances_auto(polygons_gdf, infra_gdf):
        if infra_gdf is None or len(infra_gdf) == 0:
            return np.full(len(polygons_gdf), np.nan)
        
        geom_types = infra_gdf.geometry.geom_type.unique()
        all_points = all(t == 'Point' for t in geom_types)
        
        if GPU_AVAILABLE and all_points:
            result = calculate_distances_gpu(polygons_gdf, infra_gdf)
            if result is not None:
                return result
        return calculate_distances_cpu(polygons_gdf, infra_gdf)
    
    # Calculate each distance
    print("\n  Calculating minimum distances...")
    for col_name, infra_name in distance_columns.items():
        print(f"\n    {col_name}...")
        infra_gdf = infrastructure.get(infra_name)
        if infra_gdf is None or len(infra_gdf) == 0:
            print(f"      ⚠️ No {infra_name} data, setting to NaN")
            gdf[col_name] = np.nan
            continue
        
        distances = calculate_distances_auto(gdf, infra_gdf)
        gdf[col_name] = distances
        valid = distances[~np.isnan(distances)]
        if len(valid) > 0:
            print(f"      ✅ {len(valid):,} polygons have distances")
            print(f"         Min: {valid.min():.2f}m, Mean: {valid.mean():.2f}m, Max: {valid.max():.2f}m")
        else:
            print(f"      ⚠️ No valid distances")
    
    # Save distances
    print(f"\n  Saving distances to {DISTANCES_GPKG}...")
    gdf.to_file(DISTANCES_GPKG, driver="GPKG")
    print(f"  ✅ Distances saved")

# ============================================
# STEP 2: FILTER TO THEORETICAL POTENTIAL
# ============================================

print("\n" + "="*60)
print("🔍 STEP 2: Filtering to Theoretical Potential")
print("="*60)

print("\n  Applying filters (must pass ALL):")
for col, threshold in FILTERS.items():
    print(f"    - {col} > {threshold:,}m")

# Check for required columns
missing = [col for col in FILTERS.keys() if col not in gdf.columns]
if missing:
    print(f"\n  ❌ Missing columns: {missing}")
    print("  Please ensure distances were calculated correctly.")
    exit(1)

total_polygons = len(gdf)

# Apply each filter
masks = {}
for col, threshold in FILTERS.items():
    mask = gdf[col] > threshold
    masks[col] = mask
    count_passed = mask.sum()
    pct = (count_passed / total_polygons) * 100
    print(f"\n  {col}:")
    print(f"    Threshold: > {threshold:,}m")
    print(f"    Pass: {count_passed:,} ({pct:.1f}%)")
    print(f"    Removed: {total_polygons - count_passed:,} ({100-pct:.1f}%)")

# Combine all filters (AND)
combined_mask = pd.Series(True, index=gdf.index)
for col in FILTERS.keys():
    combined_mask = combined_mask & masks[col]

filtered_gdf = gdf[combined_mask].copy()
removed_gdf = gdf[~combined_mask].copy()

kept_count = len(filtered_gdf)
removed_count = len(removed_gdf)

print(f"\n  Combined filter results:")
print(f"    Kept: {kept_count:,} ({kept_count/total_polygons*100:.1f}%)")
print(f"    Removed: {removed_count:,} ({removed_count/total_polygons*100:.1f}%)")

# ============================================
# STEP 3: SAVE THEORETICAL POTENTIAL OUTPUTS
# ============================================

print("\n=== STEP 3: Saving theoretical potential outputs ===")

# Save filtered polygons
filtered_gdf.to_file(OUTPUT_FILTERED, driver="GPKG")
print(f"  ✅ Theoretical potential polygons: {OUTPUT_FILTERED}")
print(f"     Polygons: {len(filtered_gdf):,}")

# Save removed polygons
if len(removed_gdf) > 0:
    removed_gdf.to_file(OUTPUT_REMOVED, driver="GPKG")
    print(f"  ✅ Removed polygons: {OUTPUT_REMOVED}")
    print(f"     Polygons: {len(removed_gdf):,}")

# Save statistics CSV
stats_data = []
for col, threshold in FILTERS.items():
    count_passed = masks[col].sum()
    stats_data.append({
        'filter_column': col,
        'threshold_meters': threshold,
        'polygons_passed': count_passed,
        'polygons_removed': total_polygons - count_passed,
        'percent_passed': round(count_passed/total_polygons*100, 2)
    })
stats_data.append({
    'filter_column': 'COMBINED (ALL)',
    'threshold_meters': 'N/A',
    'polygons_passed': kept_count,
    'polygons_removed': removed_count,
    'percent_passed': round(kept_count/total_polygons*100, 2)
})
stats_df = pd.DataFrame(stats_data)
stats_df.to_csv(OUTPUT_STATS, index=False)
print(f"  ✅ Statistics saved: {OUTPUT_STATS}")

# Create summary text file
with open(OUTPUT_SUMMARY, 'w') as f:
    f.write("="*60 + "\n")
    f.write("THEORETICAL POTENTIAL - FILTERING SUMMARY\n")
    f.write("="*60 + "\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Input file: {DISTANCES_GPKG}\n")
    f.write("\n")
    f.write("FILTER CRITERIA (must pass ALL):\n")
    f.write("-"*60 + "\n")
    for col, threshold in FILTERS.items():
        f.write(f"  {col} > {threshold:,}m\n")
    f.write("\n")
    f.write("RESULTS:\n")
    f.write("-"*60 + "\n")
    f.write(f"  Total polygons: {total_polygons:,}\n")
    f.write(f"  Theoretical Potential polygons: {kept_count:,} ({kept_count/total_polygons*100:.1f}%)\n")
    f.write(f"  Removed polygons: {removed_count:,} ({removed_count/total_polygons*100:.1f}%)\n")
    f.write("\n")
    
    # Area calculations
    if len(filtered_gdf) > 0:
        total_area_ha = filtered_gdf['area_ha'].sum()
        total_area_km2 = total_area_ha / 100
        f.write("AREA STATISTICS:\n")
        f.write("-"*60 + "\n")
        f.write(f"  Total theoretical potential area: {total_area_ha:,.2f} ha ({total_area_km2:,.2f} km²)\n")
        f.write("\n")
    
    # Land cover distribution
    if len(filtered_gdf) > 0 and 'class_name' in filtered_gdf.columns:
        f.write("LAND COVER DISTRIBUTION:\n")
        f.write("-"*60 + "\n")
        class_counts = filtered_gdf['class_name'].value_counts()
        for class_name, count in class_counts.items():
            pct = (count / len(filtered_gdf)) * 100
            f.write(f"  {class_name}: {count:,} polygons ({pct:.1f}%)\n")
        f.write("\n")
    
    f.write("OUTPUT FILES:\n")
    f.write("-"*60 + "\n")
    f.write(f"  Filtered polygons: {OUTPUT_FILTERED}\n")
    f.write(f"  Removed polygons: {OUTPUT_REMOVED}\n")
    f.write(f"  Statistics CSV: {OUTPUT_STATS}\n")
    f.write("="*60 + "\n")

print(f"  ✅ Summary text saved: {OUTPUT_SUMMARY}")

# ============================================
# STEP 4: DISPLAY TOTAL AREA
# ============================================

print("\n" + "="*60)
print("📊 THEORETICAL POTENTIAL - FINAL RESULTS")
print("="*60)

if len(filtered_gdf) > 0:
    total_area_ha = filtered_gdf['area_ha'].sum()
    total_area_km2 = total_area_ha / 100
    
    print(f"\n  📐 Total Theoretical Potential Area:")
    print(f"     {total_area_ha:,.2f} hectares")
    print(f"     {total_area_km2:,.2f} km²")
    print(f"     {total_area_km2 * 0.3861:,.2f} square miles")
    
    print(f"\n  📊 Summary:")
    print(f"     Total polygons: {total_polygons:,}")
    print(f"     Kept (theoretical potential): {kept_count:,} ({kept_count/total_polygons*100:.1f}%)")
    print(f"     Removed: {removed_count:,} ({removed_count/total_polygons*100:.1f}%)")
    
    print(f"\n  🌍 Land cover distribution (top 5):")
    if 'class_name' in filtered_gdf.columns:
        class_counts = filtered_gdf['class_name'].value_counts().head(5)
        for class_name, count in class_counts.items():
            pct = (count / len(filtered_gdf)) * 100
            print(f"     {class_name}: {count:,} polygons ({pct:.1f}%)")
else:
    print("\n  ⚠️ No polygons passed the filtering criteria!")

# ============================================
# FINAL SUMMARY
# ============================================

elapsed = time.time() - start_time

print("\n" + "="*60)
print("✅ PIPELINE COMPLETE!")
print("="*60)
print(f"⏱️ Total time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
print(f"🎮 GPU: {'✅ ENABLED' if GPU_AVAILABLE else '❌ DISABLED'}")
print(f"🚫 Roads: EXCLUDED (run separate script)")

print(f"\n📂 Output files:")
print(f"   📁 Distances: {DISTANCES_GPKG}")
print(f"   📁 Theoretical Potential: {THEORETICAL_DIR}")
print(f"   ├── {OUTPUT_FILTERED.name} ({kept_count:,} polygons)")
if len(removed_gdf) > 0:
    print(f"   ├── {OUTPUT_REMOVED.name} ({removed_count:,} polygons)")
print(f"   ├── {OUTPUT_STATS.name}")
print(f"   └── {OUTPUT_SUMMARY.name}")

print("\n" + "="*60)