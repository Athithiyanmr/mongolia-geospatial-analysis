"""
compare_potential_areas.py - Compare Theoretical vs Technical Potential areas by land cover class
Calculates area in km² for each land cover class and total area.
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================

# Input files
THEORETICAL_FILE = Path("outputs/vector/theoretical_potential/theoretical_potential_polygons.gpkg")
TECHNICAL_FILE = Path("outputs/vector/technical_potential_with_roads_powerlines/technical_potential_polygons.gpkg")

# Output files
OUTPUT_DIR = Path("outputs/vector/potential_comparison/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "potential_area_comparison.csv"
OUTPUT_TABLE = OUTPUT_DIR / "potential_area_table.txt"

# ============================================
# FUNCTIONS
# ============================================

def calculate_class_areas(gdf, name):
    """Calculate area in km² for each land cover class."""
    if 'class_name' not in gdf.columns:
        print(f"  ⚠️ 'class_name' column not found in {name}. Using 'class_id' instead.")
        class_col = 'class_id'
    else:
        class_col = 'class_name'
    
    # Calculate area in km²
    gdf['area_km2'] = gdf.geometry.area / 1_000_000
    
    # Group by class
    class_stats = gdf.groupby(class_col).agg({
        'area_km2': ['sum', 'count', 'mean', 'min', 'max']
    }).reset_index()
    
    class_stats.columns = [class_col, 'area_km2', 'polygon_count', 'mean_area_km2', 'min_area_km2', 'max_area_km2']
    
    # Add total row
    total_row = pd.DataFrame({
        class_col: ['TOTAL'],
        'area_km2': [class_stats['area_km2'].sum()],
        'polygon_count': [class_stats['polygon_count'].sum()],
        'mean_area_km2': [class_stats['mean_area_km2'].mean()],
        'min_area_km2': [class_stats['min_area_km2'].min()],
        'max_area_km2': [class_stats['max_area_km2'].max()]
    })
    
    class_stats = pd.concat([class_stats, total_row], ignore_index=True)
    
    return class_stats

def format_number(x):
    """Format number with commas and 2 decimal places."""
    if pd.isna(x):
        return "N/A"
    if isinstance(x, (int, float)):
        return f"{x:,.2f}"
    return str(x)

# ============================================
# MAIN
# ============================================

print("="*60)
print("📊 COMPARING THEORETICAL VS TECHNICAL POTENTIAL AREAS")
print("="*60)
print(f"\n📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Check input files
if not THEORETICAL_FILE.exists():
    print(f"  ❌ Theoretical file not found: {THEORETICAL_FILE}")
    exit(1)
if not TECHNICAL_FILE.exists():
    print(f"  ❌ Technical file not found: {TECHNICAL_FILE}")
    exit(1)

# ============================================
# STEP 1: READ THE DATA
# ============================================

print("\n=== STEP 1: Reading input files ===")

theoretical = gpd.read_file(THEORETICAL_FILE)
technical = gpd.read_file(TECHNICAL_FILE)

print(f"  Theoretical potential: {len(theoretical):,} polygons")
print(f"  Technical potential: {len(technical):,} polygons")

# ============================================
# STEP 2: CALCULATE CLASS AREAS
# ============================================

print("\n=== STEP 2: Calculating area by land cover class ===")

theoretical_stats = calculate_class_areas(theoretical, "Theoretical")
technical_stats = calculate_class_areas(technical, "Technical")

# ============================================
# STEP 3: MERGE FOR COMPARISON
# ============================================

print("\n=== STEP 3: Creating comparison table ===")

# Merge on class name
merged = theoretical_stats.merge(
    technical_stats,
    on='class_name',
    suffixes=('_theoretical', '_technical'),
    how='outer'
)

# Fill NaN with 0
merged = merged.fillna(0)

# Calculate differences
merged['area_diff_km2'] = merged['area_km2_technical'] - merged['area_km2_theoretical']
merged['area_pct_change'] = np.where(
    merged['area_km2_theoretical'] > 0,
    (merged['area_diff_km2'] / merged['area_km2_theoretical']) * 100,
    0
)

# Reorder columns for clarity
merged = merged[[
    'class_name',
    'area_km2_theoretical',
    'area_km2_technical',
    'area_diff_km2',
    'area_pct_change',
    'polygon_count_theoretical',
    'polygon_count_technical',
    'mean_area_km2_theoretical',
    'mean_area_km2_technical'
]]

# Sort by theoretical area (descending)
merged = merged.sort_values('area_km2_theoretical', ascending=False)

# ============================================
# STEP 4: SAVE OUTPUTS
# ============================================

print("\n=== STEP 4: Saving outputs ===")

# Save as CSV
merged.to_csv(OUTPUT_CSV, index=False)
print(f"  ✅ CSV saved: {OUTPUT_CSV}")

# ============================================
# STEP 5: CREATE TEXT TABLE
# ============================================

print("\n=== STEP 5: Creating formatted table ===")

def create_table(data):
    """Create a formatted text table."""
    # Header
    lines = []
    lines.append("="*120)
    lines.append("THEORETICAL VS TECHNICAL POTENTIAL - AREA COMPARISON BY LAND COVER CLASS")
    lines.append("="*120)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Summary line
    total_theoretical = data[data['class_name'] == 'TOTAL']['area_km2_theoretical'].iloc[0] if 'TOTAL' in data['class_name'].values else 0
    total_technical = data[data['class_name'] == 'TOTAL']['area_km2_technical'].iloc[0] if 'TOTAL' in data['class_name'].values else 0
    total_diff = total_technical - total_theoretical
    total_pct = (total_diff / total_theoretical * 100) if total_theoretical > 0 else 0
    
    lines.append("📊 OVERALL SUMMARY:")
    lines.append("-"*60)
    lines.append(f"  Theoretical Total Area: {total_theoretical:,.2f} km²")
    lines.append(f"  Technical Total Area:    {total_technical:,.2f} km²")
    lines.append(f"  Difference:              {total_diff:,.2f} km² ({total_pct:+.1f}%)")
    lines.append("")
    lines.append("📋 DETAILED BREAKDOWN BY CLASS:")
    lines.append("-"*120)
    
    # Header row
    header = f"{'CLASS':<25} {'THEORETICAL':>15} {'TECHNICAL':>15} {'DIFFERENCE':>15} {'CHANGE %':>10} {'COUNT (TH)':>10} {'COUNT (TECH)':>12}"
    lines.append(header)
    lines.append("-"*120)
    
    # Data rows (excluding TOTAL)
    for _, row in data.iterrows():
        if row['class_name'] == 'TOTAL':
            continue
        class_name = row['class_name'][:25]
        theo_area = row['area_km2_theoretical']
        tech_area = row['area_km2_technical']
        diff = row['area_diff_km2']
        pct = row['area_pct_change']
        count_th = row['polygon_count_theoretical']
        count_tech = row['polygon_count_technical']
        
        # Format with color indicators (using text)
        sign = "+" if diff > 0 else "" if diff == 0 else ""
        
        line = f"{class_name:<25} {theo_area:>15,.2f} {tech_area:>15,.2f} {sign}{diff:>14,.2f} {pct:>9.1f}% {count_th:>10,} {count_tech:>12,}"
        lines.append(line)
    
    # Total row
    lines.append("-"*120)
    total_row = data[data['class_name'] == 'TOTAL']
    if not total_row.empty:
        row = total_row.iloc[0]
        theo_area = row['area_km2_theoretical']
        tech_area = row['area_km2_technical']
        diff = row['area_diff_km2']
        pct = row['area_pct_change']
        sign = "+" if diff > 0 else "" if diff == 0 else ""
        lines.append(f"{'TOTAL':<25} {theo_area:>15,.2f} {tech_area:>15,.2f} {sign}{diff:>14,.2f} {pct:>9.1f}% {'':>10} {'':>12}")
    
    lines.append("="*120)
    lines.append("")
    lines.append("📝 NOTES:")
    lines.append("-"*60)
    lines.append("  - Areas are in km²")
    lines.append("  - Theoretical: Polygons meeting initial distance criteria")
    lines.append("  - Technical: Polygons meeting ALL criteria (area > 5 acres, GHI > 4.2, powerline < 10km, road < 2km)")
    lines.append("  - 'TOTAL' row includes all polygons across all classes")
    lines.append("="*120)
    
    return "\n".join(lines)

# Create and save table
table_content = create_table(merged)

with open(OUTPUT_TABLE, 'w', encoding='utf-8') as f:
    f.write(table_content)

print(f"  ✅ Formatted table saved: {OUTPUT_TABLE}")

# ============================================
# STEP 6: DISPLAY SUMMARY
# ============================================

print("\n" + "="*60)
print("📊 SUMMARY RESULTS")
print("="*60)

total_theoretical = merged[merged['class_name'] == 'TOTAL']['area_km2_theoretical'].iloc[0] if 'TOTAL' in merged['class_name'].values else 0
total_technical = merged[merged['class_name'] == 'TOTAL']['area_km2_technical'].iloc[0] if 'TOTAL' in merged['class_name'].values else 0
total_diff = total_technical - total_theoretical
total_pct = (total_diff / total_theoretical * 100) if total_theoretical > 0 else 0

print(f"\n  Theoretical Potential Area: {total_theoretical:,.2f} km²")
print(f"  Technical Potential Area:   {total_technical:,.2f} km²")
print(f"  Difference:                 {total_diff:,.2f} km² ({total_pct:+.1f}%)")

print(f"\n  Top 5 land cover classes (by theoretical area):")
theoretical_sorted = merged[merged['class_name'] != 'TOTAL'].sort_values('area_km2_theoretical', ascending=False).head(5)
for _, row in theoretical_sorted.iterrows():
    class_name = row['class_name'][:30]
    theo = row['area_km2_theoretical']
    tech = row['area_km2_technical']
    print(f"    {class_name:<30}: {theo:>10,.2f} km² (Theoretical) -> {tech:>10,.2f} km² (Technical)")

print("\n" + "="*60)
print("✅ COMPARISON COMPLETE!")
print(f"📁 Output folder: {OUTPUT_DIR}")
print(f"   ├── {OUTPUT_CSV.name}")
print(f"   └── {OUTPUT_TABLE.name}")
print("="*60)