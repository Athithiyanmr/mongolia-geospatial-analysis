"""
technical_potential_mpz_analysis.py - Extract MPZ class distribution for Technical Potential polygons only
Automatically reprojects MPZ raster to EPSG:32648 if needed.
"""

from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================
# FIX ENVIRONMENT FOR GDAL/PROJ/rasterio
# ============================================

def _fix_conda_gdal_env() -> None:
    """Point PROJ_LIB/GDAL_DATA at the active conda env, if any."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    proj_candidates = [
        Path(conda_prefix, "Library", "share", "proj"),
        Path(conda_prefix, "share", "proj"),
    ]
    for p in proj_candidates:
        if (p / "proj.db").exists():
            os.environ["PROJ_LIB"] = str(p)
            logging.info("Set PROJ_LIB to %s", p)
            break

    gdal_candidates = [
        Path(conda_prefix, "Library", "share", "gdal"),
        Path(conda_prefix, "share", "gdal"),
    ]
    for p in gdal_candidates:
        if p.exists():
            os.environ["GDAL_DATA"] = str(p)
            logging.info("Set GDAL_DATA to %s", p)
            break


logging.basicConfig(level=logging.INFO, format="%(message)s")
_fix_conda_gdal_env()

# ============================================
# IMPORTS (single pass, no duplicates)
# ============================================

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterstats import zonal_stats
from tqdm import tqdm

gpd.options.io_engine = "fiona"

# ============================================
# CONFIGURATION
# ============================================

TECHNICAL_FILE = Path("outputs/vector/technical_potential_with_roads_powerlines/technical_potential_polygons.gpkg")
MPZ_RASTER_ORIG = Path("data/raw/LULC/MPZ_v2.tif")
MPZ_RASTER_REPROJ = Path("outputs/rasters/MPZ_v2_EPSG32648.tif")

TARGET_CRS = "EPSG:32648"
TARGET_RESOLUTION_M = 30  # reprojection resolution in target CRS units

OUTPUT_DIR = Path("outputs/vector/technical_potential_with_mpz/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_GPKG = OUTPUT_DIR / "technical_potential_with_mpz.gpkg"
OUTPUT_CSV = OUTPUT_DIR / "mpz_class_distribution_technical.csv"
OUTPUT_TABLE = OUTPUT_DIR / "mpz_class_table_technical.txt"
OUTPUT_SUMMARY = OUTPUT_DIR / "mpz_summary_technical.txt"


# ============================================
# STEP 0: REPROJECT MPZ RASTER IF NEEDED
# ============================================

def reproject_raster(
    src_path: Path,
    dst_path: Path,
    target_crs: str,
    resolution: float = TARGET_RESOLUTION_M,
    resampling: Resampling = Resampling.nearest,
) -> None:
    """Reproject raster to target CRS at a fixed resolution."""
    logging.info("  Reprojecting %s to %s...", src_path, target_crs)

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds, resolution=resolution
        )
        kwargs = src.meta.copy()
        kwargs.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=resampling,
            )
    logging.info("  ✅ Reprojected raster saved to: %s", dst_path)


def resolve_mpz_raster() -> Path:
    """Return a path to an MPZ raster guaranteed to be in TARGET_CRS, reprojecting if needed."""
    if MPZ_RASTER_REPROJ.exists():
        with rasterio.open(MPZ_RASTER_REPROJ) as src:
            if str(src.crs) == TARGET_CRS:
                logging.info("  ✅ Reprojected MPZ raster already exists: %s", MPZ_RASTER_REPROJ)
                return MPZ_RASTER_REPROJ
        MPZ_RASTER_REPROJ.unlink()

    with rasterio.open(MPZ_RASTER_ORIG) as src:
        if str(src.crs) == TARGET_CRS:
            logging.info("  ✅ Original MPZ raster already in %s", TARGET_CRS)
            return MPZ_RASTER_ORIG

    reproject_raster(MPZ_RASTER_ORIG, MPZ_RASTER_REPROJ, TARGET_CRS)
    return MPZ_RASTER_REPROJ


# ============================================
# CORE EXTRACTION (rasterstats-based, windowed per feature)
# ============================================

def extract_mpz_stats(gdf: gpd.GeoDataFrame, raster_path: Path) -> tuple[pd.DataFrame, list[int]]:
    """
    Extract per-polygon MPZ class area (km²) and percentage using rasterstats.
    Each polygon only reads the raster window it overlaps — no full-raster reads in a loop.
    """
    logging.info("\n  Extracting MPZ class statistics...")

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata
        pixel_area_km2 = abs(src.res[0] * src.res[1]) / 1_000_000
        logging.info("  Pixel area: %.6f km²", pixel_area_km2)

    if gdf.crs != raster_crs:
        logging.info("  Reprojecting polygons to raster CRS (%s)...", raster_crs)
        gdf_reproj = gdf.to_crs(raster_crs)
    else:
        gdf_reproj = gdf
        logging.info("  ✅ CRS matched")

    # Drop empty/invalid geometries so zonal_stats doesn't choke on them
    valid_mask = gdf_reproj.geometry.notna() & ~gdf_reproj.geometry.is_empty
    if not valid_mask.all():
        logging.warning("  ⚠️  Skipping %d empty/invalid geometries", (~valid_mask).sum())

    stats = zonal_stats(
        gdf_reproj.geometry[valid_mask],
        str(raster_path),
        categorical=True,
        nodata=nodata,
        all_touched=True,
    )

    # Discover the full set of classes across all polygons
    class_values = sorted({int(k) for row in stats for k in row.keys()})
    logging.info("  MPZ classes found: %s", class_values)

    records = []
    for row in stats:
        total_pixels = sum(row.values())
        rec = {}
        for val in class_values:
            count = row.get(val, 0)
            rec[f"mpz_{val}_km2"] = count * pixel_area_km2
            rec[f"mpz_{val}_pct"] = (count / total_pixels * 100) if total_pixels > 0 else 0.0
        records.append(rec)

    result_df = pd.DataFrame(records, index=gdf_reproj.index[valid_mask])
    # Re-align to original index, filling skipped rows with 0
    result_df = result_df.reindex(gdf_reproj.index, fill_value=0.0)

    return result_df, class_values


def calculate_mpz_totals(gdf: gpd.GeoDataFrame, class_values: list[int]) -> pd.DataFrame:
    """Calculate total area for each MPZ class across all polygons."""
    mpz_data = []
    for val in class_values:
        km2_col, pct_col = f"mpz_{val}_km2", f"mpz_{val}_pct"
        if km2_col in gdf.columns:
            mpz_data.append(
                {
                    "mpz_class": val,
                    "total_area_km2": gdf[km2_col].sum(),
                    "avg_percentage": gdf[pct_col].mean(),
                    "polygons_with_class": int((gdf[km2_col] > 0).sum()),
                }
            )
    return pd.DataFrame(mpz_data)


def write_table(mpz_totals: pd.DataFrame, total_area_km2: float, n_polygons: int, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TECHNICAL POTENTIAL - MPZ CLASS DISTRIBUTION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Total polygons: {n_polygons:,}\n")
        f.write(f"Total area: {total_area_km2:,.2f} km²\n\n")
        f.write(f"{'MPZ CLASS':<12} {'AREA (km²)':>18} {'% OF TOTAL':>15} {'POLYGONS WITH CLASS':>20}\n")
        f.write("-" * 80 + "\n")
        for _, row in mpz_totals.iterrows():
            pct_of_total = (row["total_area_km2"] / total_area_km2 * 100) if total_area_km2 > 0 else 0
            f.write(
                f"{int(row['mpz_class']):<12} {row['total_area_km2']:>18,.2f} "
                f"{pct_of_total:>14.1f}% {row['polygons_with_class']:>20,}\n"
            )
        f.write("-" * 80 + "\n")
        f.write(f"{'TOTAL':<12} {total_area_km2:>18,.2f} {100.0:>14.1f}% {n_polygons:>20,}\n")
        f.write("=" * 80 + "\n\n")
        f.write("📝 NOTES:\n" + "-" * 80 + "\n")
        f.write("  - Areas are in km²\n")
        f.write("  - MPZ classes represent different land use/cover categories\n")
        f.write("  - '% OF TOTAL' shows the percentage of the total Technical Potential area\n")
        f.write("  - 'POLYGONS WITH CLASS' counts how many polygons contain each MPZ class\n")
        f.write("=" * 80 + "\n")


def write_summary(mpz_totals: pd.DataFrame, total_area_km2: float, n_polygons: int, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\nTECHNICAL POTENTIAL - MPZ SUMMARY\n" + "=" * 60 + "\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write(f"Total Technical Potential Polygons: {n_polygons:,}\n")
        f.write(f"Total Technical Potential Area: {total_area_km2:,.2f} km²\n\n")
        f.write("MPZ Class Distribution:\n" + "-" * 60 + "\n")
        for _, row in mpz_totals.iterrows():
            pct = (row["total_area_km2"] / total_area_km2 * 100) if total_area_km2 > 0 else 0
            f.write(f"  MPZ {int(row['mpz_class'])}: {row['total_area_km2']:>12,.2f} km² ({pct:>5.1f}%)\n")
        f.write("=" * 60 + "\n")


# ============================================
# MAIN
# ============================================

def main() -> None:
    logging.info("=" * 60)
    logging.info("📊 TECHNICAL POTENTIAL - MPZ CLASS ANALYSIS")
    logging.info("=" * 60)
    logging.info("📅 Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if not TECHNICAL_FILE.exists():
        raise FileNotFoundError(f"Technical file not found: {TECHNICAL_FILE}")

    mpz_raster = resolve_mpz_raster()
    if not mpz_raster.exists():
        raise FileNotFoundError(f"MPZ raster not found: {mpz_raster}")
    logging.info("  MPZ raster found: %s", mpz_raster)

    logging.info("\n=== STEP 1: Reading Technical Potential ===")
    gdf = gpd.read_file(TECHNICAL_FILE, engine="fiona")
    logging.info("  Technical potential polygons: %s", f"{len(gdf):,}")
    logging.info("  CRS: %s", gdf.crs)

    logging.info("\n=== STEP 2: Extracting MPZ statistics ===")
    mpz_df, mpz_classes = extract_mpz_stats(gdf, mpz_raster)
    for col in mpz_df.columns:
        gdf[col] = mpz_df[col].values
    logging.info("  ✅ MPZ statistics extracted for %s polygons", f"{len(gdf):,}")
    logging.info("  ✅ %d MPZ classes found: %s", len(mpz_classes), mpz_classes)

    logging.info("\n=== STEP 3: Saving outputs ===")
    gdf.to_file(OUTPUT_GPKG, driver="GPKG")
    logging.info("  ✅ GeoPackage saved: %s", OUTPUT_GPKG)

    mpz_totals = calculate_mpz_totals(gdf, mpz_classes)
    mpz_totals.to_csv(OUTPUT_CSV, index=False)
    logging.info("  ✅ MPZ totals CSV saved: %s", OUTPUT_CSV)

    logging.info("\n=== STEP 4: Creating formatted table ===")
    total_area_km2 = gdf.geometry.area.sum() / 1_000_000
    write_table(mpz_totals, total_area_km2, len(gdf), OUTPUT_TABLE)
    logging.info("  ✅ Formatted table saved: %s", OUTPUT_TABLE)

    logging.info("\n=== STEP 5: Creating summary ===")
    write_summary(mpz_totals, total_area_km2, len(gdf), OUTPUT_SUMMARY)
    logging.info("  ✅ Summary saved: %s", OUTPUT_SUMMARY)

    logging.info("\n" + "=" * 60)
    logging.info("📊 MPZ CLASS DISTRIBUTION - TECHNICAL POTENTIAL")
    logging.info("=" * 60)
    logging.info("  Total Technical Potential Area: %.2f km²", total_area_km2)
    logging.info("  Total Polygons: %s", f"{len(gdf):,}")
    logging.info("  MPZ Classes: %d\n", len(mpz_classes))
    for _, row in mpz_totals.iterrows():
        pct = (row["total_area_km2"] / total_area_km2 * 100) if total_area_km2 > 0 else 0
        logging.info("    MPZ %d: %12.2f km² (%5.1f%%)", int(row["mpz_class"]), row["total_area_km2"], pct)

    logging.info("\n" + "=" * 60)
    logging.info("✅ TECHNICAL POTENTIAL MPZ ANALYSIS COMPLETE!")
    logging.info("📁 Output folder: %s", OUTPUT_DIR)
    logging.info("   ├── %s", OUTPUT_GPKG.name)
    logging.info("   ├── %s", OUTPUT_CSV.name)
    logging.info("   ├── %s", OUTPUT_TABLE.name)
    logging.info("   └── %s", OUTPUT_SUMMARY.name)
    logging.info("=" * 60)


if __name__ == "__main__":
    main()