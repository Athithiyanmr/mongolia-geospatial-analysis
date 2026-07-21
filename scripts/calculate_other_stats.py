"""
calculate_infrastructure_distances_minimal.py - Calculate ONLY minimum distances
Minimal version - only calculates min_dist columns (no categories, no counts)
EXCLUDES: roads (too large, run separately if needed)
Includes: highways, powerlines, protected lands, railways, substations, airports, waterbodies
NO SEARCH BUFFER - finds nearest infrastructure regardless of distance
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

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
OUTPUT_GPKG = Path("outputs/vector/grassland_dry_steppe_with_distances.gpkg")
OUTPUT_DIR = Path("outputs/vector/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- CONFIGURATION ---
WATER_CLASS_VALUE = 0  # Water class in LULC
# NO SEARCH BUFFER - will search all infrastructure
# --------------------------------

print("\n" + "="*60)
print("🎯 MINIMUM DISTANCES ONLY")
print("🚫 EXCLUDING: Roads (run separately)")
print("✅ INCLUDING: Highways, Powerlines, Protected, Railways, Substations, Airports, Waterbodies")
print("🔍 NO SEARCH BUFFER - finds nearest infrastructure regardless of distance")
print("="*60)
start_time = time.time()

# ============================================
# STEP 1: READ EXISTING POLYGONS
# ============================================

print("\n=== STEP 1: Reading polygons ===")
gdf = gpd.read_file(INPUT_POLYGONS)
print(f"  Loaded {len(gdf):,} polygons")
print(f"  CRS: {gdf.crs}")

# ============================================
# STEP 2: READ INFRASTRUCTURE DATA (NO ROADS)
# ============================================

print("\n=== STEP 2: Reading infrastructure (roads excluded) ===")

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
                
                print(f"  ✅ {name}: {len(gdf_infra):,}")
                return gdf_infra
            return None
        else:
            print(f"  ⚠️ {name} not found")
            return None
    except Exception as e:
        print(f"  ❌ Error loading {name}: {e}")
        return None

# Read all infrastructure (NO ROADS)
infrastructure = {}
infrastructure['highways'] = read_infrastructure(HIGHWAYS, "Highways")
infrastructure['powerlines'] = read_infrastructure(POWERLINES, "Powerlines")
infrastructure['protected'] = read_infrastructure(PROTECTED_LANDS, "Protected Lands")
infrastructure['railways'] = read_infrastructure(RAILWAYS, "Railways")
infrastructure['substations'] = read_infrastructure(SUBSTATIONS, "Substations")
infrastructure['airports'] = read_infrastructure(AIRPORTS, "Airports")

# Count total features
total_features = sum([len(v) if v is not None else 0 for v in infrastructure.values()])
print(f"\n  Total infrastructure features loaded: {total_features:,}")

# ============================================
# STEP 3: EXTRACT WATERBODIES FROM LULC
# ============================================

print("\n=== STEP 3: Extracting waterbodies from LULC ===")

try:
    import rasterio
    from rasterio import features
    from shapely.geometry import shape
    import numpy as np
    
    if LULC_FILLED_PATH.exists():
        print(f"  Reading: {LULC_FILLED_PATH}")
        
        with rasterio.open(LULC_FILLED_PATH) as src:
            lulc_data = src.read(1)
            
            print(f"  Raster shape: {lulc_data.shape}")
            print(f"  Raster dtype: {lulc_data.dtype}")
            
            # Create water mask
            if lulc_data.dtype in ['float64', 'float32']:
                water_mask = np.isclose(lulc_data, float(WATER_CLASS_VALUE))
            else:
                water_mask = (lulc_data == WATER_CLASS_VALUE)
            
            water_pixel_count = int(water_mask.sum())
            print(f"  Water pixels: {water_pixel_count:,}")
            
            if water_pixel_count > 0:
                print("  Vectorizing water pixels...")
                water_mask_uint8 = water_mask.astype(np.uint8)
                shapes_gen = features.shapes(water_mask_uint8, mask=water_mask_uint8, transform=src.transform)
                
                water_features = []
                for geom, value in tqdm(shapes_gen, desc="    Creating water polygons"):
                    if value == 1:
                        try:
                            polygon = shape(geom)
                            if polygon.is_valid and not polygon.is_empty:
                                water_features.append({'geometry': polygon})
                        except:
                            continue
                
                if water_features:
                    water_gdf = gpd.GeoDataFrame(water_features, crs=src.crs)
                    print(f"  Created {len(water_gdf):,} water polygons")
                    
                    if water_gdf.crs != gdf.crs:
                        print(f"  Reprojecting to {gdf.crs}...")
                        water_gdf = water_gdf.to_crs(gdf.crs)
                    
                    # Use centroids for distance calculation
                    water_gdf['geometry'] = water_gdf.geometry.centroid
                    water_gdf = water_gdf.set_geometry('geometry')
                    
                    infrastructure['waterbodies'] = water_gdf
                    print(f"  ✅ Waterbodies: {len(water_gdf):,}")
                else:
                    print("  ⚠️ No water features extracted")
            else:
                print("  ⚠️ No water pixels found")
    else:
        print(f"  ⚠️ LULC file not found: {LULC_FILLED_PATH}")
        
except Exception as e:
    print(f"  ⚠️ Error extracting waterbodies: {e}")

# ============================================
# STEP 4: DISTANCE CALCULATION (NO SEARCH BUFFER)
# ============================================

def calculate_distances_gpu(polygons_gdf, infra_gdf, batch_size=50000):
    """
    GPU-accelerated distance calculation using PyTorch
    """
    try:
        print("    🚀 Using GPU...")
        
        # Get polygon centroids
        centroids = polygons_gdf.geometry.centroid
        polygon_coords = np.array([[p.x, p.y] for p in centroids], dtype=np.float32)
        infra_coords = np.array([[p.x, p.y] for p in infra_gdf.geometry], dtype=np.float32)
        
        print(f"      Polygons: {polygon_coords.shape}, Infrastructure: {infra_coords.shape}")
        
        # Move to GPU
        polygon_coords_gpu = torch.tensor(polygon_coords, device='cuda')
        infra_coords_gpu = torch.tensor(infra_coords, device='cuda')
        
        all_distances = []
        n_polygons = len(polygon_coords_gpu)
        
        # Process in batches
        for i in tqdm(range(0, n_polygons, batch_size), desc="      GPU batch"):
            end_idx = min(i + batch_size, n_polygons)
            batch_polygons = polygon_coords_gpu[i:end_idx]
            
            for j in range(len(batch_polygons)):
                point = batch_polygons[j]
                diff = infra_coords_gpu - point
                dist = torch.sqrt(torch.sum(diff * diff, dim=1))
                min_dist = torch.min(dist)
                all_distances.append(min_dist.item())
        
        # Clean up
        del polygon_coords_gpu, infra_coords_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return np.array(all_distances)
        
    except Exception as e:
        print(f"    ⚠️ GPU failed: {e}")
        return None

def calculate_distances_cpu(polygons_gdf, infra_gdf):
    """
    CPU distance calculation using spatial index (NO SEARCH BUFFER)
    Finds nearest infrastructure regardless of distance
    """
    print("    💻 Using CPU (no search buffer)...")
    
    distances = []
    sindex = infra_gdf.sindex
    geometries = infra_gdf.geometry
    centroids = polygons_gdf.geometry.centroid
    
    for idx, row in tqdm(polygons_gdf.iterrows(), total=len(polygons_gdf), desc="      Calculating"):
        polygon = row.geometry
        centroid = centroids.iloc[idx]
        
        # Get all possible matches using polygon bounds (no buffer)
        minx, miny, maxx, maxy = polygon.bounds
        possible = list(sindex.intersection((minx, miny, maxx, maxy)))
        
        # If no matches, use centroid with large buffer
        if not possible:
            # Use a 100km buffer as fallback
            buffered = centroid.buffer(100000)  # 100km
            possible = list(sindex.intersection(buffered.bounds))
        
        # If still no matches, search all (slow but ensures no NaN)
        if not possible:
            possible = list(range(len(geometries)))
        
        candidates = geometries.iloc[possible]
        
        try:
            # Distance to polygon (works for LineString too)
            min_dist = candidates.distance(polygon).min()
            
            # Check if intersects
            if candidates.intersects(polygon).any():
                min_dist = 0.0
            distances.append(min_dist)
        except:
            # Fallback to centroid
            try:
                min_dist = candidates.distance(centroid).min()
                distances.append(min_dist)
            except:
                distances.append(np.nan)
    
    return np.array(distances)

def calculate_distances_auto(polygons_gdf, infra_gdf):
    """
    Auto-select GPU or CPU based on availability and geometry type
    """
    if infra_gdf is None or len(infra_gdf) == 0:
        return np.full(len(polygons_gdf), np.nan)
    
    # Check if all geometries are points (GPU works best)
    geom_types = infra_gdf.geometry.geom_type.unique()
    all_points = all(t == 'Point' for t in geom_types)
    
    if GPU_AVAILABLE and all_points:
        # Try GPU
        result = calculate_distances_gpu(polygons_gdf, infra_gdf)
        if result is not None:
            return result
    
    # Fallback to CPU
    return calculate_distances_cpu(polygons_gdf, infra_gdf)

# ============================================
# STEP 5: CALCULATE ALL MINIMUM DISTANCES
# ============================================

print("\n=== STEP 5: Calculating minimum distances (no search buffer) ===")

# Define which columns to calculate (NO ROADS)
distance_columns = {
    'min_dist_highway': 'highways',
    'min_dist_powerline': 'powerlines',
    'min_dist_protected': 'protected',
    'min_dist_railway': 'railways',
    'min_dist_substation': 'substations',
    'min_dist_airport': 'airports',
    'min_dist_waterbody': 'waterbodies'
}

# Dictionary to store stats for summary
stats_summary = {}

# Calculate each distance
for col_name, infra_name in distance_columns.items():
    print(f"\n  Calculating {col_name}...")
    
    infra_gdf = infrastructure.get(infra_name)
    if infra_gdf is None or len(infra_gdf) == 0:
        print(f"    ⚠️ No {infra_name} data, setting to NaN")
        gdf[col_name] = np.nan
        stats_summary[col_name] = {'valid': 0, 'min': np.nan, 'mean': np.nan, 'max': np.nan}
        continue
    
    # Calculate distances (auto-select GPU/CPU)
    distances = calculate_distances_auto(gdf, infra_gdf)
    gdf[col_name] = distances
    
    # Statistics
    valid = distances[~np.isnan(distances)]
    if len(valid) > 0:
        print(f"    ✅ {len(valid):,} polygons have distances")
        print(f"       Min: {valid.min():.2f} m ({valid.min()/1000:.2f} km)")
        print(f"       Mean: {valid.mean():.2f} m ({valid.mean()/1000:.2f} km)")
        print(f"       Max: {valid.max():.2f} m ({valid.max()/1000:.2f} km)")
        
        stats_summary[col_name] = {
            'valid': len(valid),
            'min': valid.min(),
            'mean': valid.mean(),
            'max': valid.max()
        }
    else:
        print(f"    ⚠️ No valid distances calculated")
        stats_summary[col_name] = {'valid': 0, 'min': np.nan, 'mean': np.nan, 'max': np.nan}

# ============================================
# STEP 6: SAVE OUTPUT
# ============================================

print("\n=== STEP 6: Saving output ===")

# Save
gdf.to_file(OUTPUT_GPKG, driver="GPKG")
print(f"  ✅ Saved: {OUTPUT_GPKG}")

# ============================================
# SUMMARY
# ============================================

elapsed = time.time() - start_time

print("\n" + "="*60)
print("⚡ SUMMARY")
print("="*60)
print(f"⏱️ Time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
print(f"📊 Polygons: {len(gdf):,}")
print(f"📊 Distance columns added: {len(distance_columns)}")
print(f"🎮 GPU: {'✅ ENABLED' if GPU_AVAILABLE else '❌ DISABLED'}")
print(f"🚫 Roads: EXCLUDED (run separate script)")
print(f"🔍 Search Buffer: NONE (finds nearest regardless of distance)")

print("\n📂 Output:")
print(f"   {OUTPUT_GPKG}")

print("\n✅ Distance columns added:")
for col in distance_columns:
    stats = stats_summary.get(col, {})
    valid = stats.get('valid', 0)
    pct = (valid / len(gdf) * 100) if len(gdf) > 0 else 0
    print(f"   ✅ {col}: {valid:,} polygons ({pct:.1f}%)")

print("\n📝 Distance statistics:")
for col in distance_columns:
    stats = stats_summary.get(col, {})
    if stats.get('valid', 0) > 0:
        min_km = stats['min'] / 1000
        mean_km = stats['mean'] / 1000
        max_km = stats['max'] / 1000
        print(f"   {col.replace('min_dist_', '')}:")
        print(f"      Min: {stats['min']:.0f}m ({min_km:.1f}km)")
        print(f"      Mean: {stats['mean']:.0f}m ({mean_km:.1f}km)")
        print(f"      Max: {stats['max']:.0f}m ({max_km:.1f}km)")

print("\n📝 Columns included:")
print("   - min_dist_highway")
print("   - min_dist_powerline")
print("   - min_dist_protected")
print("   - min_dist_railway")
print("   - min_dist_substation")
print("   - min_dist_airport")
print("   - min_dist_waterbody")

print("\n💡 To add roads, run a separate script for roads only")
print("💡 Large distances (>50km) are normal in Mongolia's sparse landscape")

print("="*60)