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

# Output clipped + projected raster
OUTPUT_RASTER = Path(
    "outputs/rasters/"
    "LULC_2025_AOI_EPSG32648.tif"
)

# Output statistics CSV
OUTPUT_CSV = Path(
    "outputs/stats/"
    "landcover_stats_AOI_epsg32648.csv"
)

# Output AOI summary CSV
OUTPUT_AOI_SUMMARY_CSV = Path(
    "outputs/stats/"
    "aoi_area_summary_epsg32648.csv"
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
    255: "NoData",
}


# ============================================================
# CONSTANTS
# ============================================================

# Original raster NoData value that must be measured
SOURCE_NODATA_VALUE = 255

# Outside AOI value in final raster.
#
# We must NOT use 255 because 255 inside AOI
# needs to be counted separately.
#
# Therefore output raster will use UInt16,
# allowing 65535 as outside-AOI NoData.
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

    # Remove null geometry
    aoi = aoi[
        aoi.geometry.notnull()
    ].copy()

    # Remove empty geometry
    aoi = aoi[
        ~aoi.geometry.is_empty
    ].copy()

    if aoi.empty:
        raise ValueError(
            "AOI contains no usable geometries."
        )

    # --------------------------------------------------------
    # Fix invalid geometry
    # --------------------------------------------------------

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

    print(
        f"AOI path: {aoi_path}"
    )

    print(
        f"AOI CRS: {aoi.crs}"
    )

    print(
        f"AOI feature count: {len(aoi)}"
    )

    return aoi


# ============================================================
# STEP 2: CALCULATE VECTOR AOI AREA
# ============================================================

def calculate_aoi_area(
    aoi: gpd.GeoDataFrame,
) -> dict:

    """
    Calculate exact AOI polygon area
    after projection to EPSG:32648.
    """

    # --------------------------------------------------------
    # Reproject AOI
    # --------------------------------------------------------

    aoi_projected = aoi.to_crs(
        AREA_CRS
    )

    # --------------------------------------------------------
    # Dissolve all features
    # --------------------------------------------------------

    aoi_union = (
        aoi_projected
        .geometry
        .union_all()
    )

    # --------------------------------------------------------
    # Calculate area
    # --------------------------------------------------------

    area_m2 = float(
        aoi_union.area
    )

    area_ha = (
        area_m2
        / 10_000
    )

    area_acres = (
        area_m2
        / 4046.8564224
    )

    area_km2 = (
        area_m2
        / 1_000_000
    )

    print("\nAOI Vector Area")
    print("=" * 70)

    print(
        f"Area CRS: "
        f"{AREA_CRS.to_string()}"
    )

    print(
        f"AOI area: "
        f"{area_m2:,.2f} m²"
    )

    print(
        f"AOI area: "
        f"{area_ha:,.2f} hectares"
    )

    print(
        f"AOI area: "
        f"{area_acres:,.2f} acres"
    )

    print(
        f"AOI area: "
        f"{area_km2:,.4f} km²"
    )

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

    Critical logic:

    - Original 255 inside AOI is preserved
    - Outside AOI becomes 65535
    - Output dtype becomes uint16

    This allows us to distinguish:

    255   = original NoData inside AOI
    65535 = outside AOI
    """

    with rasterio.open(
        raster_path
    ) as src:

        if src.crs is None:
            raise ValueError(
                "Raster has no CRS."
            )

        print("\nOriginal Raster Information")
        print("=" * 70)

        print(
            f"Raster: {raster_path}"
        )

        print(
            f"CRS: {src.crs}"
        )

        print(
            f"Resolution: {src.res}"
        )

        print(
            f"Width: {src.width}"
        )

        print(
            f"Height: {src.height}"
        )

        print(
            f"Metadata NoData: {src.nodata}"
        )

        print(
            f"Bounds: {src.bounds}"
        )

        # ----------------------------------------------------
        # AOI to raster CRS
        # ----------------------------------------------------

        aoi_raster_crs = aoi.to_crs(
            src.crs
        )

        # ----------------------------------------------------
        # Union geometry
        # ----------------------------------------------------

        aoi_geometry = [
            aoi_raster_crs
            .geometry
            .union_all()
        ]

        # ----------------------------------------------------
        # Read clipped raster as MASKED ARRAY
        # ----------------------------------------------------

        clipped_masked, clipped_transform = mask(
            dataset=src,
            shapes=aoi_geometry,

            crop=True,

            # IMPORTANT:
            # Return MaskedArray
            filled=False,

            all_touched=False,
        )

        # ----------------------------------------------------
        # Extract first band
        # ----------------------------------------------------

        band = clipped_masked[0]

        # ----------------------------------------------------
        # Convert to uint16
        # ----------------------------------------------------

        clipped_data = np.full(
            band.shape,
            OUTSIDE_AOI_NODATA,
            dtype=np.uint16,
        )

        # ----------------------------------------------------
        # Inside AOI mask
        # ----------------------------------------------------

        inside_aoi_mask = (
            ~np.ma.getmaskarray(
                band
            )
        )

        # ----------------------------------------------------
        # Copy all original values inside AOI
        #
        # This includes:
        # 0-9
        # 255
        # ----------------------------------------------------

        clipped_data[
            inside_aoi_mask
        ] = (
            band.data[
                inside_aoi_mask
            ]
            .astype(np.uint16)
        )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        inside_pixel_count = int(
            inside_aoi_mask.sum()
        )

        nodata_255_count = int(
            np.sum(
                inside_aoi_mask
                & (
                    clipped_data
                    == SOURCE_NODATA_VALUE
                )
            )
        )

        print("\nClipped Raster Diagnostics")
        print("=" * 70)

        print(
            f"Inside-AOI raster pixels: "
            f"{inside_pixel_count:,}"
        )

        print(
            f"255 pixels inside AOI: "
            f"{nodata_255_count:,}"
        )

        print(
            f"Outside AOI value: "
            f"{OUTSIDE_AOI_NODATA}"
        )

        # ----------------------------------------------------
        # Output profile
        # ----------------------------------------------------

        profile = src.profile.copy()

        profile.update(
            {
                "height": clipped_data.shape[0],
                "width": clipped_data.shape[1],
                "transform": clipped_transform,

                # UInt16 required for 65535
                "dtype": "uint16",

                "count": 1,

                # Only outside AOI is raster NoData
                "nodata": OUTSIDE_AOI_NODATA,
            }
        )

        # ----------------------------------------------------
        # Save to memory
        # ----------------------------------------------------

        memfile = MemoryFile()

        with memfile.open(
            **profile
        ) as dst:

            dst.write(
                clipped_data,
                1,
            )

        return memfile


# ============================================================
# STEP 4: REPROJECT TO EPSG:32648
# ============================================================

def reproject_clipped_raster(
    clipped_memfile: MemoryFile,
) -> MemoryFile:

    """
    Reproject clipped categorical raster
    to EPSG:32648.

    Preserves:
    - classes 0-9
    - inside-AOI 255

    Outside AOI:
    - 65535
    """

    with clipped_memfile.open() as src:

        transform, width, height = (
            calculate_default_transform(
                src.crs,
                AREA_CRS,
                src.width,
                src.height,
                *src.bounds,
            )
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

        with projected_memfile.open(
            **profile
        ) as dst:

            reproject(
                source=rasterio.band(
                    src,
                    1,
                ),

                destination=rasterio.band(
                    dst,
                    1,
                ),

                src_transform=src.transform,
                src_crs=src.crs,

                # Only 65535 means outside AOI
                src_nodata=OUTSIDE_AOI_NODATA,

                dst_transform=transform,
                dst_crs=AREA_CRS,

                dst_nodata=OUTSIDE_AOI_NODATA,

                # Categorical raster
                resampling=Resampling.nearest,
            )

        return projected_memfile


# ============================================================
# STEP 5: CALCULATE RASTER STATISTICS
# ============================================================

def calculate_raster_stats(
    projected_memfile: MemoryFile,
) -> pd.DataFrame:

    """
    Calculate class areas.

    Includes:
    - classes 0-9
    - 255 NoData inside AOI

    Excludes:
    - 65535 outside AOI
    """

    with projected_memfile.open() as src:

        raster = src.read(1)

        transform = src.transform

        # ----------------------------------------------------
        # Pixel area
        # ----------------------------------------------------

        pixel_width = abs(
            transform.a
        )

        pixel_height = abs(
            transform.e
        )

        pixel_area_m2 = (
            pixel_width
            * pixel_height
        )

        # ----------------------------------------------------
        # Inside AOI
        #
        # Everything except 65535
        # ----------------------------------------------------

        inside_aoi_mask = (
            raster
            != OUTSIDE_AOI_NODATA
        )

        inside_data = raster[
            inside_aoi_mask
        ]

        if inside_data.size == 0:
            raise ValueError(
                "No pixels found inside AOI."
            )

        # ----------------------------------------------------
        # Unique values
        # ----------------------------------------------------

        unique_values, counts = np.unique(
            inside_data,
            return_counts=True,
        )

        # ----------------------------------------------------
        # Build statistics
        # ----------------------------------------------------

        stats = pd.DataFrame(
            {
                "value": unique_values,
                "pixel_count": counts,
            }
        )

        stats["class_name"] = (
            stats["value"]
            .map(LULC_CLASSES)
            .fillna("Unknown")
        )

        # ----------------------------------------------------
        # Area calculations
        # ----------------------------------------------------

        stats["area_m2"] = (
            stats["pixel_count"]
            * pixel_area_m2
        )

        stats["area_ha"] = (
            stats["area_m2"]
            / 10_000
        )

        stats["area_acres"] = (
            stats["area_m2"]
            / 4046.8564224
        )

        stats["area_km2"] = (
            stats["area_m2"]
            / 1_000_000
        )

        # ----------------------------------------------------
        # Percentage of rasterized AOI
        # ----------------------------------------------------

        rasterized_aoi_area_m2 = (
            stats["area_m2"].sum()
        )

        stats["percentage_of_rasterized_aoi"] = (
            stats["area_m2"]
            / rasterized_aoi_area_m2
            * 100
        )

        # ----------------------------------------------------
        # Column order
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Print raster information
        # ----------------------------------------------------

        print("\nProjected Raster Information")
        print("=" * 70)

        print(
            f"CRS: {src.crs}"
        )

        print(
            f"Resolution: {src.res}"
        )

        print(
            f"Pixel area: "
            f"{pixel_area_m2:.4f} m²"
        )

        print(
            f"Inside-AOI pixel count: "
            f"{inside_data.size:,}"
        )

        return stats


# ============================================================
# STEP 6: SAVE PROJECTED RASTER
# ============================================================

def save_memory_raster(
    memfile: MemoryFile,
    output_path: Path,
):

    """
    Save projected raster.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with memfile.open() as src:

        profile = src.profile.copy()

        profile.update(
            {
                "driver": "GTiff",
                "compress": "lzw",
            }
        )

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as dst:

            dst.write(
                src.read()
            )

    print("\nRaster saved to:")
    print(output_path)


# ============================================================
# STEP 7: CREATE AOI SUMMARY
# ============================================================

def create_aoi_summary(
    stats: pd.DataFrame,
    vector_aoi_area: dict,
) -> pd.DataFrame:

    """
    Create summary showing:

    - exact vector AOI area
    - rasterized AOI area
    - classified area
    - 255 NoData area
    """

    # --------------------------------------------------------
    # Rasterized AOI area
    # ----------------------------------------------------

    rasterized_aoi_area_km2 = (
        stats["area_km2"].sum()
    )

    # --------------------------------------------------------
    # 255 NoData
    # ----------------------------------------------------

    nodata_rows = stats[
        stats["value"]
        == SOURCE_NODATA_VALUE
    ]

    if nodata_rows.empty:

        nodata_255_area_km2 = 0.0
        nodata_255_pixels = 0

    else:

        nodata_255_area_km2 = float(
            nodata_rows[
                "area_km2"
            ].iloc[0]
        )

        nodata_255_pixels = int(
            nodata_rows[
                "pixel_count"
            ].iloc[0]
        )

    # --------------------------------------------------------
    # Classified area
    # ----------------------------------------------------

    classified_rows = stats[
        stats["value"].isin(
            list(range(10))
        )
    ]

    classified_area_km2 = (
        classified_rows[
            "area_km2"
        ].sum()
    )

    # --------------------------------------------------------
    # Percentages
    # ----------------------------------------------------

    rasterized_total = (
        rasterized_aoi_area_km2
    )

    if rasterized_total > 0:

        nodata_percentage = (
            nodata_255_area_km2
            / rasterized_total
            * 100
        )

        classified_percentage = (
            classified_area_km2
            / rasterized_total
            * 100
        )

    else:

        nodata_percentage = 0.0
        classified_percentage = 0.0

    # --------------------------------------------------------
    # Build summary
    # ----------------------------------------------------

    summary = pd.DataFrame(
        {
            "metric": [
                "Exact vector AOI area",
                "Rasterized AOI area",
                "Classified LULC area",
                "NoData 255 area",
            ],

            "area_km2": [
                vector_aoi_area[
                    "aoi_area_km2"
                ],

                rasterized_aoi_area_km2,

                classified_area_km2,

                nodata_255_area_km2,
            ],

            "pixel_count": [
                np.nan,

                int(
                    stats[
                        "pixel_count"
                    ].sum()
                ),

                int(
                    classified_rows[
                        "pixel_count"
                    ].sum()
                ),

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

    # --------------------------------------------------------
    # Create folders
    # --------------------------------------------------------

    OUTPUT_RASTER.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_AOI_SUMMARY_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clipped_memfile = None
    projected_memfile = None

    try:

        # ====================================================
        # 1. Read AOI
        # ====================================================

        aoi = read_aoi(
            AOI_PATH
        )

        # ====================================================
        # 2. Calculate exact AOI area
        # ====================================================

        vector_aoi_area = (
            calculate_aoi_area(
                aoi
            )
        )

        # ====================================================
        # 3. Clip raster
        # ====================================================

        clipped_memfile = (
            clip_raster_to_aoi(
                raster_path=RASTER_PATH,
                aoi=aoi,
            )
        )

        # ====================================================
        # 4. Reproject to EPSG:32648
        # ====================================================

        projected_memfile = (
            reproject_clipped_raster(
                clipped_memfile
            )
        )

        # ====================================================
        # 5. Save raster
        # ====================================================

        save_memory_raster(
            memfile=projected_memfile,
            output_path=OUTPUT_RASTER,
        )

        # ====================================================
        # 6. Calculate statistics
        # ====================================================

        stats = calculate_raster_stats(
            projected_memfile
        )

        # ====================================================
        # 7. Create AOI summary
        # ====================================================

        summary = create_aoi_summary(
            stats=stats,
            vector_aoi_area=vector_aoi_area,
        )

        # ====================================================
        # 8. Round outputs
        # ====================================================

        area_columns = [
            "area_m2",
            "area_ha",
            "area_acres",
        ]

        for column in area_columns:

            stats[column] = (
                stats[column]
                .round(2)
            )

        stats["area_km2"] = (
            stats["area_km2"]
            .round(4)
        )

        stats[
            "percentage_of_rasterized_aoi"
        ] = (
            stats[
                "percentage_of_rasterized_aoi"
            ]
            .round(2)
        )

        summary["area_km2"] = (
            summary["area_km2"]
            .round(4)
        )

        summary[
            "percentage_of_rasterized_aoi"
        ] = (
            summary[
                "percentage_of_rasterized_aoi"
            ]
            .round(2)
        )

        # ====================================================
        # 9. Print LULC statistics
        # ====================================================

        print("\nLULC + NoData Statistics")
        print("=" * 130)

        print(
            stats.to_string(
                index=False
            )
        )

        # ====================================================
        # 10. Print AOI summary
        # ====================================================

        print("\nAOI Area Summary")
        print("=" * 100)

        print(
            summary.to_string(
                index=False
            )
        )

        # ====================================================
        # 11. Save CSVs
        # ====================================================

        stats.to_csv(
            OUTPUT_CSV,
            index=False,
        )

        summary.to_csv(
            OUTPUT_AOI_SUMMARY_CSV,
            index=False,
        )

        print("\nOutputs saved:")
        print(OUTPUT_RASTER)
        print(OUTPUT_CSV)
        print(OUTPUT_AOI_SUMMARY_CSV)

    finally:

        if projected_memfile is not None:
            projected_memfile.close()

        if clipped_memfile is not None:
            clipped_memfile.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()