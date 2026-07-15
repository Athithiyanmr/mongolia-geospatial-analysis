from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)
from pyproj import CRS
from scipy.ndimage import distance_transform_edt


# ============================================================
# CONFIGURATION
# ============================================================

# Input LULC raster
RASTER_PATH = Path(
    "data/raw/LULC/LULC_2025_WGS84.tif"
)

# AOI boundary
AOI_PATH = Path(
    "data/raw/aoi/aoi_mn.shp"
)

# Output clipped + projected raster (RAW, with 255 still present)
OUTPUT_RASTER_RAW = Path(
    "outputs/rasters/"
    "LULC_2025_AOI_EPSG32648_raw.tif"
)

# Output clipped + projected + FILLED raster (no more 255)
OUTPUT_RASTER_FILLED = Path(
    "outputs/rasters/"
    "LULC_2025_AOI_EPSG32648_filled.tif"
)

# Output statistics CSV (computed on the FILLED raster)
OUTPUT_CSV = Path(
    "outputs/stats/"
    "landcover_stats_AOI_epsg32648_filled.csv"
)

# Output AOI summary CSV
OUTPUT_AOI_SUMMARY_CSV = Path(
    "outputs/stats/"
    "aoi_area_summary_epsg32648_filled.csv"
)


# ============================================================
# LULC CLASS MAPPING
# ============================================================

LULC_CLASSES = {
    0: "Water",
    1: "Forest",
    2: "Grassland",
    3: "Dry steppe",
    4: "Barren",
    5: "Built-up",
    6: "Cropland",
    7: "Roads",
    8: "Sand",
    9: "Wetland",

    # IMPORTANT:
    # Keep 255 as a reportable class
    # (should be 0 pixels after fill step)
    255: "NoData",
}


# ============================================================
# CONSTANTS
# ============================================================

# Original raster NoData value that must be measured / filled
SOURCE_NODATA_VALUE = 255

# Outside AOI value in final raster.
OUTSIDE_AOI_NODATA = 65535

# Area CRS
AREA_CRS = CRS.from_epsg(32648)


# ============================================================
# STEP 1: READ AND PREPARE AOI
# ============================================================

def read_aoi(
    aoi_path: Path,
) -> gpd.GeoDataFrame:

    """
    Read, clean and validate AOI.
    """

    print("\nReading AOI")
    print("=" * 70)

    aoi = gpd.read_file(
        aoi_path
    )

    if aoi.empty:
        raise ValueError(
            "AOI contains no features."
        )

    if aoi.crs is None:
        raise ValueError(
            "AOI has no CRS."
        )

    aoi = aoi[
        aoi.geometry.notnull()
    ].copy()

    aoi = aoi[
        ~aoi.geometry.is_empty
    ].copy()

    if aoi.empty:
        raise ValueError(
            "AOI contains no usable geometries."
        )

    invalid_count = int(
        (~aoi.geometry.is_valid).sum()
    )

    if invalid_count > 0:

        print(
            f"Fixing {invalid_count} "
            f"invalid geometries..."
        )

        aoi["geometry"] = (
            aoi.geometry.buffer(0)
        )

    print(f"AOI path: {aoi_path}")
    print(f"AOI CRS: {aoi.crs}")
    print(f"AOI feature count: {len(aoi)}")

    return aoi


# ============================================================
# STEP 2: CALCULATE VECTOR AOI AREA
# ============================================================

def calculate_aoi_area(
    aoi: gpd.GeoDataFrame,
) -> dict:

    """
    Calculate exact AOI polygon area after projection to EPSG:32648.
    """

    aoi_projected = aoi.to_crs(AREA_CRS)
    aoi_union = aoi_projected.geometry.union_all()
    area_m2 = float(aoi_union.area)
    area_ha = area_m2 / 10_000
    area_acres = area_m2 / 4046.8564224
    area_km2 = area_m2 / 1_000_000

    print("\nAOI Vector Area")
    print("=" * 70)
    print(f"Area CRS: {AREA_CRS.to_string()}")
    print(f"AOI area: {area_m2:,.2f} m2")
    print(f"AOI area: {area_ha:,.2f} hectares")
    print(f"AOI area: {area_acres:,.2f} acres")
    print(f"AOI area: {area_km2:,.4f} km2")

    return {
        "aoi_area_m2": area_m2,
        "aoi_area_ha": area_ha,
        "aoi_area_acres": area_acres,
        "aoi_area_km2": area_km2,
    }


# ============================================================
# STEP 3: CLIP RASTER TO AOI
# ============================================================

def clip_raster_to_aoi(
    raster_path: Path,
    aoi: gpd.GeoDataFrame,
) -> MemoryFile:

    """
    Clip raster to AOI.

    - Original 255 inside AOI is preserved
    - Outside AOI becomes 65535
    - Output dtype becomes uint16
    """

    with rasterio.open(raster_path) as src:

        if src.crs is None:
            raise ValueError("Raster has no CRS.")

        print("\nOriginal Raster Information")
        print("=" * 70)
        print(f"Raster: {raster_path}")
        print(f"CRS: {src.crs}")
        print(f"Resolution: {src.res}")
        print(f"Width: {src.width}")
        print(f"Height: {src.height}")
        print(f"Metadata NoData: {src.nodata}")
        print(f"Bounds: {src.bounds}")

        aoi_raster_crs = aoi.to_crs(src.crs)
        aoi_geometry = [aoi_raster_crs.geometry.union_all()]

        clipped_masked, clipped_transform = mask(
            dataset=src,
            shapes=aoi_geometry,
            crop=True,
            filled=False,
            all_touched=False,
        )

        band = clipped_masked[0]

        clipped_data = np.full(
            band.shape,
            OUTSIDE_AOI_NODATA,
            dtype=np.uint16,
        )

        inside_aoi_mask = ~np.ma.getmaskarray(band)

        clipped_data[inside_aoi_mask] = (
            band.data[inside_aoi_mask].astype(np.uint16)
        )

        inside_pixel_count = int(inside_aoi_mask.sum())
        nodata_255_count = int(
            np.sum(
                inside_aoi_mask
                & (clipped_data == SOURCE_NODATA_VALUE)
            )
        )

        print("\nClipped Raster Diagnostics")
        print("=" * 70)
        print(f"Inside-AOI raster pixels: {inside_pixel_count:,}")
        print(f"255 pixels inside AOI: {nodata_255_count:,}")
        print(f"Outside AOI value: {OUTSIDE_AOI_NODATA}")

        profile = src.profile.copy()

        profile.update(
            {
                "height": clipped_data.shape[0],
                "width": clipped_data.shape[1],
                "transform": clipped_transform,
                "dtype": "uint16",
                "count": 1,
                "nodata": OUTSIDE_AOI_NODATA,
            }
        )

        memfile = MemoryFile()

        with memfile.open(**profile) as dst:
            dst.write(clipped_data, 1)

        return memfile


# ============================================================
# STEP 4: REPROJECT TO EPSG:32648
# ============================================================

def reproject_clipped_raster(
    clipped_memfile: MemoryFile,
) -> MemoryFile:

    """
    Reproject clipped categorical raster to EPSG:32648.
    Preserves classes 0-9, inside-AOI 255, outside-AOI 65535.
    """

    with clipped_memfile.open() as src:

        transform, width, height = calculate_default_transform(
            src.crs,
            AREA_CRS,
            src.width,
            src.height,
            *src.bounds,
        )

        profile = src.profile.copy()

        profile.update(
            {
                "crs": AREA_CRS,
                "transform": transform,
                "width": width,
                "height": height,
                "dtype": "uint16",
                "nodata": OUTSIDE_AOI_NODATA,
            }
        )

        projected_memfile = MemoryFile()

        with projected_memfile.open(**profile) as dst:

            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=OUTSIDE_AOI_NODATA,
                dst_transform=transform,
                dst_crs=AREA_CRS,
                dst_nodata=OUTSIDE_AOI_NODATA,
                resampling=Resampling.nearest,
            )

        return projected_memfile


# ============================================================
# STEP 5 (NEW): FILL 255 NODATA WITH NEAREST VALID NEIGHBOR
# ============================================================

def fill_nodata_nearest_neighbor(
    projected_memfile: MemoryFile,
) -> MemoryFile:

    """
    Fill pixels equal to SOURCE_NODATA_VALUE (255) using the
    value of the nearest valid class pixel (0-9).

    Rules:
    - Only pixels == 255 get overwritten.
    - Pixels == OUTSIDE_AOI_NODATA (65535) are NEVER used as a
      fill source and are NEVER overwritten. They stay exactly
      as they are (outside the AOI).
    - This is true nearest-neighbor "copy the closest valid
      pixel" filling, appropriate for categorical/class data
      (unlike IDW-style fillnodata, which can blend classes).
    """

    with projected_memfile.open() as src:
        data = src.read(1)
        profile = src.profile.copy()

    invalid_mask = (data == SOURCE_NODATA_VALUE)

    # Valid = a real class value (not 255, not 65535)
    valid_mask = (
        (data != SOURCE_NODATA_VALUE)
        & (data != OUTSIDE_AOI_NODATA)
    )

    total_invalid = int(invalid_mask.sum())

    print("\nNoData Fill (nearest neighbor)")
    print("=" * 70)
    print(f"255 pixels to fill: {total_invalid:,}")

    if total_invalid == 0:
        print("No 255 pixels found — nothing to fill.")
        filled_memfile = MemoryFile()
        with filled_memfile.open(**profile) as dst:
            dst.write(data, 1)
        return filled_memfile

    if not valid_mask.any():
        raise ValueError(
            "No valid class pixels available inside AOI to fill from."
        )

    # For every pixel, find the index of the nearest True (valid)
    # pixel in valid_mask.
    nearest_row_idx, nearest_col_idx = distance_transform_edt(
        ~valid_mask,
        return_distances=False,
        return_indices=True,
    )

    filled_source = data[nearest_row_idx, nearest_col_idx]

    output_data = data.copy()
    output_data[invalid_mask] = filled_source[invalid_mask]

    # Sanity check: outside-AOI pixels must be untouched
    outside_mask = (data == OUTSIDE_AOI_NODATA)
    assert np.array_equal(
        output_data[outside_mask], data[outside_mask]
    ), "Outside-AOI pixels were modified — this should never happen."

    remaining_255 = int(
        np.sum(output_data == SOURCE_NODATA_VALUE)
    )
    print(f"255 pixels remaining after fill: {remaining_255:,}")

    filled_memfile = MemoryFile()

    with filled_memfile.open(**profile) as dst:
        dst.write(output_data, 1)

    return filled_memfile


# ============================================================
# STEP 6: CALCULATE RASTER STATISTICS
# ============================================================

def calculate_raster_stats(
    projected_memfile: MemoryFile,
) -> pd.DataFrame:

    """
    Calculate class areas. Excludes 65535 (outside AOI).
    """

    with projected_memfile.open() as src:

        raster = src.read(1)
        transform = src.transform

        pixel_width = abs(transform.a)
        pixel_height = abs(transform.e)
        pixel_area_m2 = pixel_width * pixel_height

        inside_aoi_mask = (raster != OUTSIDE_AOI_NODATA)
        inside_data = raster[inside_aoi_mask]

        if inside_data.size == 0:
            raise ValueError("No pixels found inside AOI.")

        unique_values, counts = np.unique(
            inside_data, return_counts=True
        )

        stats = pd.DataFrame(
            {"value": unique_values, "pixel_count": counts}
        )

        stats["class_name"] = (
            stats["value"].map(LULC_CLASSES).fillna("Unknown")
        )

        stats["area_m2"] = stats["pixel_count"] * pixel_area_m2
        stats["area_ha"] = stats["area_m2"] / 10_000
        stats["area_acres"] = stats["area_m2"] / 4046.8564224
        stats["area_km2"] = stats["area_m2"] / 1_000_000

        rasterized_aoi_area_m2 = stats["area_m2"].sum()

        stats["percentage_of_rasterized_aoi"] = (
            stats["area_m2"] / rasterized_aoi_area_m2 * 100
        )

        stats = stats[
            [
                "value",
                "class_name",
                "pixel_count",
                "area_m2",
                "area_ha",
                "area_acres",
                "area_km2",
                "percentage_of_rasterized_aoi",
            ]
        ]

        print("\nProjected Raster Information")
        print("=" * 70)
        print(f"CRS: {src.crs}")
        print(f"Resolution: {src.res}")
        print(f"Pixel area: {pixel_area_m2:.4f} m2")
        print(f"Inside-AOI pixel count: {inside_data.size:,}")

        return stats


# ============================================================
# STEP 7: SAVE MEMORY RASTER
# ============================================================

def save_memory_raster(
    memfile: MemoryFile,
    output_path: Path,
):

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with memfile.open() as src:

        profile = src.profile.copy()
        profile.update({"driver": "GTiff", "compress": "lzw"})

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(src.read())

    print("\nRaster saved to:")
    print(output_path)


# ============================================================
# STEP 8: CREATE AOI SUMMARY
# ============================================================

def create_aoi_summary(
    stats: pd.DataFrame,
    vector_aoi_area: dict,
) -> pd.DataFrame:

    rasterized_aoi_area_km2 = stats["area_km2"].sum()

    nodata_rows = stats[stats["value"] == SOURCE_NODATA_VALUE]

    if nodata_rows.empty:
        nodata_255_area_km2 = 0.0
        nodata_255_pixels = 0
    else:
        nodata_255_area_km2 = float(nodata_rows["area_km2"].iloc[0])
        nodata_255_pixels = int(nodata_rows["pixel_count"].iloc[0])

    classified_rows = stats[stats["value"].isin(list(range(10)))]
    classified_area_km2 = classified_rows["area_km2"].sum()

    rasterized_total = rasterized_aoi_area_km2

    if rasterized_total > 0:
        nodata_percentage = nodata_255_area_km2 / rasterized_total * 100
        classified_percentage = classified_area_km2 / rasterized_total * 100
    else:
        nodata_percentage = 0.0
        classified_percentage = 0.0

    summary = pd.DataFrame(
        {
            "metric": [
                "Exact vector AOI area",
                "Rasterized AOI area",
                "Classified LULC area",
                "NoData 255 area (after fill)",
            ],
            "area_km2": [
                vector_aoi_area["aoi_area_km2"],
                rasterized_aoi_area_km2,
                classified_area_km2,
                nodata_255_area_km2,
            ],
            "pixel_count": [
                np.nan,
                int(stats["pixel_count"].sum()),
                int(classified_rows["pixel_count"].sum()),
                nodata_255_pixels,
            ],
            "percentage_of_rasterized_aoi": [
                np.nan,
                100.0,
                classified_percentage,
                nodata_percentage,
            ],
        }
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_RASTER_RAW.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RASTER_FILLED.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_AOI_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    clipped_memfile = None
    projected_memfile = None
    filled_memfile = None

    try:

        aoi = read_aoi(AOI_PATH)
        vector_aoi_area = calculate_aoi_area(aoi)

        clipped_memfile = clip_raster_to_aoi(
            raster_path=RASTER_PATH, aoi=aoi
        )

        projected_memfile = reproject_clipped_raster(clipped_memfile)

        # Save the RAW (pre-fill) reprojected raster too, for reference
        save_memory_raster(
            memfile=projected_memfile, output_path=OUTPUT_RASTER_RAW
        )

        # NEW STEP: fill 255 pixels with nearest valid neighbor
        filled_memfile = fill_nodata_nearest_neighbor(projected_memfile)

        # Save the FILLED raster (this is your final deliverable)
        save_memory_raster(
            memfile=filled_memfile, output_path=OUTPUT_RASTER_FILLED
        )

        # Stats now computed on the FILLED raster
        stats = calculate_raster_stats(filled_memfile)
        summary = create_aoi_summary(
            stats=stats, vector_aoi_area=vector_aoi_area
        )

        area_columns = ["area_m2", "area_ha", "area_acres"]
        for column in area_columns:
            stats[column] = stats[column].round(2)

        stats["area_km2"] = stats["area_km2"].round(4)
        stats["percentage_of_rasterized_aoi"] = (
            stats["percentage_of_rasterized_aoi"].round(2)
        )

        summary["area_km2"] = summary["area_km2"].round(4)
        summary["percentage_of_rasterized_aoi"] = (
            summary["percentage_of_rasterized_aoi"].round(2)
        )

        print("\nLULC + NoData Statistics (post-fill)")
        print("=" * 130)
        print(stats.to_string(index=False))

        print("\nAOI Area Summary (post-fill)")
        print("=" * 100)
        print(summary.to_string(index=False))

        stats.to_csv(OUTPUT_CSV, index=False)
        summary.to_csv(OUTPUT_AOI_SUMMARY_CSV, index=False)

        print("\nOutputs saved:")
        print(OUTPUT_RASTER_RAW)
        print(OUTPUT_RASTER_FILLED)
        print(OUTPUT_CSV)
        print(OUTPUT_AOI_SUMMARY_CSV)

    finally:

        if filled_memfile is not None:
            filled_memfile.close()

        if projected_memfile is not None:
            projected_memfile.close()

        if clipped_memfile is not None:
            clipped_memfile.close()


if __name__ == "__main__":
    main()