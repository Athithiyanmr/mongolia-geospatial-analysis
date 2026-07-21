"""
filter_technical_potential_with_roads.py - Filter theoretical potential polygons to technical potential
Applies additional criteria:
- Area > 5 acres (converted from hectares)
- Mean GHI > 4.2
- Distance to substation < 10 km (10,000 m)
- Distance to road < 2 km (2,000 m)

Input: theoretical_potential_polygons.gpkg (from previous step)
Output: technical_potential_without_roads/
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================

# Input: theoretical potential polygons
INPUT_GPKG = Path("outputs/vector/theoretical_potential/theoretical_potential_polygons.gpkg")

# Roads data
ROADS_SHP = Path("data/raw/aoi_clipped/road/road_clipped.shp")

# Output directory - TECHNICAL POTENTIAL (without roads)
OUTPUT_DIR = Path("outputs/vector/technical_potential_without_roads/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output files
OUTPUT_FILTERED = OUTPUT_DIR / "technical_potential_polygons.gpkg"
OUTPUT_STATS = OUTPUT_DIR / "filtering_statistics.csv"
OUTPUT_REMOVED = OUTPUT_DIR / "removed_polygons.gpkg"
OUTPUT_SUMMARY = OUTPUT_DIR / "filtering_summary.txt"

# ============================================
# FILTERING CRITERIA
# ============================================

# Define filtering thresholds
# Area: 5 acres = 2.02343 hectares (since 1 acre = 0.404686 ha)
ACRES_TO_HA = 0.404686
MIN_AREA_HA = 5 * ACRES_TO_HA  # ~2.02343 ha

FILTERS = {
    'area_ha': MIN_AREA_HA,           # > 5 acres (in hectares)
    'mean_ghi': 4.2,                  # > 4.2
    'min_dist_substation': 10000,     # < 10,000m (10 km)
    'min_dist_road': 2000             # < 2,000m (2 km)  <--- NEW FILTER
}

# Filter descriptions for output
FILTER_DESCRIPTIONS = {
    'area_ha': f'Area > 5 acres ({MIN_AREA_HA:.4f} ha)',
    'mean_ghi': 'Mean GHI > 4.2',
    'min_dist_substation': 'Distance to substation < 10 km',
    'min_dist_road': 'Distance to road < 2 km'
}

print("="*60)
print("🔍 FILTERING TO TECHNICAL POTENTIAL (without roads)")
print("📁 Output: technical_potential_without_roads/")
print("="*60)
print(f"\n📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================
# STEP 1: READ THE INPUT DATA (Theoretical Potential)
# ============================================

print("\n=== STEP 1: Reading theoretical potential data ===")

if not INPUT_GPKG.exists():
    print(f"  ❌ Input file not found: {INPUT_GPKG}")
    print("  Please run the theoretical potential filtering first.")
    exit(1)

gdf = gpd.read_file(INPUT_GPKG)
print(f"  Loaded {len(gdf):,} polygons")
print(f"  CRS: {gdf.crs}")

# ============================================
# STEP 2: CALCULATE MINIMUM DISTANCE TO ROADS
# ============================================

print("\n=== STEP 2: Calculating minimum distance to roads ===")

if not ROADS_SHP.exists():
    print(f"  ⚠️ Roads file not found: {ROADS_SHP}")
    print("  Skipping road distance calculation.")
    gdf['min_dist_road'] = np.nan
else:
    print(f"  Reading roads: {ROADS_SHP}")
    roads = gpd.read_file(ROADS_SHP)
    print(f"  Loaded {len(roads):,} road features")

    # Reproject roads if needed
    if roads.crs != gdf.crs:
        print(f"  Reprojecting roads to {gdf.crs}...")
        roads = roads.to_crs(gdf.crs)

    # Build spatial index
    sindex = roads.sindex
    geometries = roads.geometry

    print("  Calculating minimum distances to roads (no search buffer)...")
    distances = []
    centroids = gdf.geometry.centroid

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="    Computing road distances"):
        polygon = row.geometry
        centroid = centroids.iloc[idx]

        # Use polygon bounds to query spatial index (no buffer)
        minx, miny, maxx, maxy = polygon.bounds
        possible = list(sindex.intersection((minx, miny, maxx, maxy)))

        # Fallback: if no candidates, use a 100km buffer around centroid
        if not possible:
            buffered = centroid.buffer(100000)  # 100 km
            possible = list(sindex.intersection(buffered.bounds))

        # If still no candidates, search all (should be rare)
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

    gdf['min_dist_road'] = distances
    valid = gdf['min_dist_road'].dropna()
    if len(valid) > 0:
        print(f"  ✅ Road distances calculated for {len(valid):,} polygons")
        print(f"     Min: {valid.min():.2f}m, Mean: {valid.mean():.2f}m, Max: {valid.max():.2f}m")
    else:
        print("  ⚠️ No valid road distances calculated")

# ============================================
# STEP 3: CHECK REQUIRED COLUMNS FOR FILTERING
# ============================================

print("\n=== STEP 3: Checking required columns for filtering ===")

required_columns = ['area_ha', 'mean_ghi', 'min_dist_substation', 'min_dist_road']
missing = [col for col in required_columns if col not in gdf.columns]

if missing:
    print(f"  ❌ Missing columns: {missing}")
    print("  Please ensure theoretical potential data has these columns.")
    exit(1)
else:
    print("  ✅ All required columns found")

# ============================================
# STEP 4: APPLY TECHNICAL FILTERS
# ============================================

print("\n=== STEP 4: Applying technical filters ===")
print("  Filter criteria (must pass ALL):")
for col, threshold in FILTERS.items():
    desc = FILTER_DESCRIPTIONS.get(col, f'{col} > {threshold}')
    # For substation and road, condition is LESS THAN; for others GREATER THAN
    if 'substation' in col or 'road' in col:
        print(f"    - {desc}")
    else:
        print(f"    - {desc}")

# Create masks
masks = {}
total_polygons = len(gdf)

# Area filter: area_ha > MIN_AREA_HA
mask_area = gdf['area_ha'] > MIN_AREA_HA
masks['area_ha'] = mask_area
count_area = mask_area.sum()
print(f"\n  Area ({MIN_AREA_HA:.4f} ha):")
print(f"    Pass: {count_area:,} ({count_area/total_polygons*100:.1f}%)")
print(f"    Removed: {total_polygons - count_area:,} ({100-count_area/total_polygons*100:.1f}%)")

# GHI filter: mean_ghi > 4.2
mask_ghi = gdf['mean_ghi'] > 4.2
masks['mean_ghi'] = mask_ghi
count_ghi = mask_ghi.sum()
print(f"\n  GHI (> 4.2):")
print(f"    Pass: {count_ghi:,} ({count_ghi/total_polygons*100:.1f}%)")
print(f"    Removed: {total_polygons - count_ghi:,} ({100-count_ghi/total_polygons*100:.1f}%)")

# Substation distance filter: min_dist_substation < 10000
mask_sub = gdf['min_dist_substation'] < 10000
masks['min_dist_substation'] = mask_sub
count_sub = mask_sub.sum()
print(f"\n  Substation distance (< 10,000m):")
print(f"    Pass: {count_sub:,} ({count_sub/total_polygons*100:.1f}%)")
print(f"    Removed: {total_polygons - count_sub:,} ({100-count_sub/total_polygons*100:.1f}%)")

# Road distance filter: min_dist_road < 2000
mask_road = gdf['min_dist_road'] < 2000
masks['min_dist_road'] = mask_road
count_road = mask_road.sum()
print(f"\n  Road distance (< 2,000m):")
print(f"    Pass: {count_road:,} ({count_road/total_polygons*100:.1f}%)")
print(f"    Removed: {total_polygons - count_road:,} ({100-count_road/total_polygons*100:.1f}%)")

# Combine all filters (AND condition)
print("\n  Combining all filters (must pass ALL criteria)...")
combined_mask = mask_area & mask_ghi & mask_sub & mask_road

filtered_gdf = gdf[combined_mask].copy()
removed_gdf = gdf[~combined_mask].copy()

# ============================================
# STEP 5: FILTER STATISTICS
# ============================================

kept_count = len(filtered_gdf)
removed_count = len(removed_gdf)

print(f"\n  📊 Summary:")
print(f"    Total theoretical potential polygons: {total_polygons:,}")
print(f"    ✅ Technical Potential (kept): {kept_count:,} ({kept_count/total_polygons*100:.1f}%)")
print(f"    ❌ Removed: {removed_count:,} ({removed_count/total_polygons*100:.1f}%)")

# ============================================
# STEP 6: SAVE RESULTS
# ============================================

print("\n=== STEP 6: Saving results to technical_potential_without_roads/ ===")

# Save filtered polygons
filtered_gdf.to_file(OUTPUT_FILTERED, driver="GPKG")
print(f"  ✅ Technical Potential polygons: {OUTPUT_FILTERED}")
print(f"     Polygons: {len(filtered_gdf):,}")

# Save removed polygons
if len(removed_gdf) > 0:
    removed_gdf.to_file(OUTPUT_REMOVED, driver="GPKG")
    print(f"  ✅ Removed polygons: {OUTPUT_REMOVED}")
    print(f"     Polygons: {len(removed_gdf):,}")

# ============================================
# STEP 7: CREATE STATISTICS CSV
# ============================================

print("\n=== STEP 7: Creating statistics ===")

stats_data = []
for col, threshold in FILTERS.items():
    count_passed = masks[col].sum()
    count_removed = total_polygons - count_passed
    desc = FILTER_DESCRIPTIONS.get(col, col)
    stats_data.append({
        'filter_criteria': desc,
        'column': col,
        'threshold': threshold,
        'polygons_passed': count_passed,
        'polygons_removed': count_removed,
        'percent_passed': round(count_passed/total_polygons*100, 2),
        'percent_removed': round((total_polygons - count_passed)/total_polygons*100, 2)
    })

# Combined
stats_data.append({
    'filter_criteria': 'COMBINED (ALL) - Technical Potential',
    'column': 'N/A',
    'threshold': 'N/A',
    'polygons_passed': kept_count,
    'polygons_removed': removed_count,
    'percent_passed': round(kept_count/total_polygons*100, 2),
    'percent_removed': round(removed_count/total_polygons*100, 2)
})

stats_df = pd.DataFrame(stats_data)
stats_df.to_csv(OUTPUT_STATS, index=False)
print(f"  ✅ Statistics saved: {OUTPUT_STATS}")

# ============================================
# STEP 8: CREATE SUMMARY TEXT FILE
# ============================================

print("\n=== STEP 8: Creating summary text file ===")

with open(OUTPUT_SUMMARY, 'w') as f:
    f.write("="*60 + "\n")
    f.write("TECHNICAL POTENTIAL (without roads) - FILTERING SUMMARY\n")
    f.write("="*60 + "\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Input file: {INPUT_GPKG}\n")
    f.write("\n")
    
    f.write("FILTER CRITERIA (must pass ALL):\n")
    f.write("-"*60 + "\n")
    for col, threshold in FILTERS.items():
        desc = FILTER_DESCRIPTIONS.get(col, f'{col} > {threshold}')
        f.write(f"  {desc}\n")
    f.write("\n")
    
    f.write("RESULTS:\n")
    f.write("-"*60 + "\n")
    f.write(f"  Total theoretical potential polygons: {total_polygons:,}\n")
    f.write(f"  Technical Potential polygons: {kept_count:,} ({kept_count/total_polygons*100:.1f}%)\n")
    f.write(f"  Removed polygons: {removed_count:,} ({removed_count/total_polygons*100:.1f}%)\n")
    f.write("\n")
    
    # Area statistics for technical potential
    if len(filtered_gdf) > 0:
        total_area_ha = filtered_gdf['area_ha'].sum()
        total_area_km2 = total_area_ha / 100
        total_area_acres = total_area_ha / 0.404686
        f.write("AREA STATISTICS:\n")
        f.write("-"*60 + "\n")
        f.write(f"  Total technical potential area: {total_area_ha:,.2f} ha\n")
        f.write(f"  Total technical potential area: {total_area_km2:,.2f} km²\n")
        f.write(f"  Total technical potential area: {total_area_acres:,.2f} acres\n")
        f.write("\n")
    
    # GHI statistics
    if len(filtered_gdf) > 0 and 'mean_ghi' in filtered_gdf.columns:
        f.write("GHI STATISTICS:\n")
        f.write("-"*60 + "\n")
        f.write(f"  Mean GHI: {filtered_gdf['mean_ghi'].mean():.4f}\n")
        f.write(f"  Min GHI: {filtered_gdf['mean_ghi'].min():.4f}\n")
        f.write(f"  Max GHI: {filtered_gdf['mean_ghi'].max():.4f}\n")
        f.write("\n")
    
    # Substation distance statistics
    if len(filtered_gdf) > 0 and 'min_dist_substation' in filtered_gdf.columns:
        f.write("SUBSTATION DISTANCE STATISTICS:\n")
        f.write("-"*60 + "\n")
        f.write(f"  Mean distance: {filtered_gdf['min_dist_substation'].mean():.0f} m\n")
        f.write(f"  Min distance: {filtered_gdf['min_dist_substation'].min():.0f} m\n")
        f.write(f"  Max distance: {filtered_gdf['min_dist_substation'].max():.0f} m\n")
        f.write("\n")
    
    # Road distance statistics
    if len(filtered_gdf) > 0 and 'min_dist_road' in filtered_gdf.columns:
        road_valid = filtered_gdf['min_dist_road'].dropna()
        if len(road_valid) > 0:
            f.write("ROAD DISTANCE STATISTICS:\n")
            f.write("-"*60 + "\n")
            f.write(f"  Mean distance: {road_valid.mean():.0f} m\n")
            f.write(f"  Min distance: {road_valid.min():.0f} m\n")
            f.write(f"  Max distance: {road_valid.max():.0f} m\n")
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
# STEP 9: SUMMARY OF TECHNICAL POTENTIAL POLYGONS
# ============================================

print("\n=== STEP 9: Summary of technical potential polygons ===")

if len(filtered_gdf) > 0:
    total_area_ha = filtered_gdf['area_ha'].sum()
    total_area_km2 = total_area_ha / 100
    total_area_acres = total_area_ha / 0.404686
    
    print(f"\n  📐 Area:")
    print(f"    Total technical potential area: {total_area_ha:,.2f} ha")
    print(f"    Total technical potential area: {total_area_km2:,.2f} km²")
    print(f"    Total technical potential area: {total_area_acres:,.2f} acres")
    
    print(f"\n  ☀️ GHI (mean):")
    print(f"    Mean: {filtered_gdf['mean_ghi'].mean():.4f}")
    print(f"    Min: {filtered_gdf['mean_ghi'].min():.4f}")
    print(f"    Max: {filtered_gdf['mean_ghi'].max():.4f}")
    
    print(f"\n  ⚡ Substation distance (meters):")
    print(f"    Mean: {filtered_gdf['min_dist_substation'].mean():.0f} m")
    print(f"    Min: {filtered_gdf['min_dist_substation'].min():.0f} m")
    print(f"    Max: {filtered_gdf['min_dist_substation'].max():.0f} m")
    
    if 'min_dist_road' in filtered_gdf.columns:
        road_valid = filtered_gdf['min_dist_road'].dropna()
        if len(road_valid) > 0:
            print(f"\n  🚗 Road distance (meters):")
            print(f"    Mean: {road_valid.mean():.0f} m")
            print(f"    Min: {road_valid.min():.0f} m")
            print(f"    Max: {road_valid.max():.0f} m")
    
    if 'class_name' in filtered_gdf.columns:
        print(f"\n  🌍 Land cover distribution (top 5):")
        class_counts = filtered_gdf['class_name'].value_counts().head(5)
        for class_name, count in class_counts.items():
            pct = (count / len(filtered_gdf)) * 100
            print(f"    {class_name}: {count:,} polygons ({pct:.1f}%)")
else:
    print("\n  ⚠️ No polygons passed the technical filtering criteria!")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "="*60)
print("✅ TECHNICAL POTENTIAL FILTERING COMPLETE!")
print("📁 Results saved in: technical_potential_without_roads/")
print("="*60)

print(f"\n📂 Output files:")
print(f"   📁 {OUTPUT_DIR}")
print(f"   ├── {OUTPUT_FILTERED.name} ({kept_count:,} polygons - TECHNICAL POTENTIAL)")
if len(removed_gdf) > 0:
    print(f"   ├── {OUTPUT_REMOVED.name} ({removed_count:,} polygons - REMOVED)")
print(f"   ├── {OUTPUT_STATS.name} (statistics)")
print(f"   └── {OUTPUT_SUMMARY.name} (summary report)")

print(f"\n📊 Summary:")
print(f"   Total theoretical potential polygons: {total_polygons:,}")
print(f"   ✅ Technical Potential: {kept_count:,} ({kept_count/total_polygons*100:.1f}%)")
print(f"   ❌ Removed: {removed_count:,} ({removed_count/total_polygons*100:.1f}%)")

print("\n🔍 Filter conditions (must pass ALL):")
for col, threshold in FILTERS.items():
    desc = FILTER_DESCRIPTIONS.get(col, f'{col} > {threshold}')
    print(f"   - {desc}")

if len(filtered_gdf) > 0:
    print(f"\n🏆 Technical Potential Area:")
    print(f"   {total_area_km2:.2f} km²")
    print(f"   {total_area_acres:,.2f} acres")

print("\n" + "="*60)