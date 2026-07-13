from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from rasterio.warp import (
    reproject,
    Resampling,
)


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Input clipped + projected LULC raster
# Output from your previous script
# ------------------------------------------------------------

LULC_PATH = Path(
    "outputs/rasters/"
    "LULC_2025_AOI_EPSG32648.tif"
)


# ------------------------------------------------------------
# Input slope raster
# Expected approximately 30 m resolution
# ------------------------------------------------------------

SLOPE_PATH = Path(
    "data/raw/terrain/"
    "tov_ulaanbaatur_slope_30m.tif"
)


# ------------------------------------------------------------
# Input elevation raster / DEM
# Expected approximately 30 m resolution
# ------------------------------------------------------------

ELEVATION_PATH = Path(
    "data/raw/terrain/"
    "tov_ulaanbaatur_elevation_30m.tif"
)


# ------------------------------------------------------------
# Output pixel-level CSV
# One row per selected LULC pixel
# ------------------------------------------------------------

OUTPUT_PIXEL_CSV = Path(
    "outputs/terrain/"
    "grassland_dry_steppe_pixel_values.csv"
)


# ------------------------------------------------------------
# Output summary statistics
# ------------------------------------------------------------

OUTPUT_SUMMARY_CSV = Path(
    "outputs/terrain/"
    "grassland_dry_steppe_summary_stats.csv"
)


# ------------------------------------------------------------
# Output combined raster
#
# Band 1 = LULC class
# Band 2 = Slope
# Band 3 = Elevation
# ------------------------------------------------------------

OUTPUT_RASTER = Path(
    "outputs/terrain/"
    "grassland_dry_steppe_slope_elevation.tif"
)


# ============================================================
# TARGET LULC CLASSES
# ============================================================

TARGET_CLASSES = {
    2: "Grassland",
    3: "Dry steppe",
}


# ============================================================
# OUTPUT NODATA
# ============================================================

FLOAT_NODATA = -9999.0


# ============================================================
# HELPER: VALID DATA MASK
# ============================================================

def get_valid_mask(
    data: np.ndarray,
    nodata,
) -> np.ndarray:

    """
    Create valid-data mask for a raster array.
    """

    valid = np.ones(
        data.shape,
        dtype=bool,
    )

    # --------------------------------------------------------
    # Handle raster NoData
    # --------------------------------------------------------

    if nodata is not None:

        if (
            np.issubdtype(
                data.dtype,
                np.floating,
            )
            and np.isnan(nodata)
        ):

            valid &= ~np.isnan(data)

        else:

            valid &= data != nodata

    # --------------------------------------------------------
    # Extra NaN protection
    # --------------------------------------------------------

    if np.issubdtype(
        data.dtype,
        np.floating,
    ):

        valid &= ~np.isnan(data)

    return valid


# ============================================================
# PRINT RASTER INFORMATION
# ============================================================

def print_raster_info(
    raster_path: Path,
    label: str,
):

    """
    Print raster metadata for checking CRS,
    resolution, dimensions and NoData.
    """

    with rasterio.open(
        raster_path
    ) as src:

        print(f"\n{label}")
        print("=" * 70)

        print(
            f"Path: {raster_path}"
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
            f"NoData: {src.nodata}"
        )

        print(
            f"Bounds: {src.bounds}"
        )

        print(
            f"Data type: {src.dtypes[0]}"
        )


# ============================================================
# ALIGN CONTINUOUS RASTER TO LULC GRID
# ============================================================

def align_continuous_raster_to_lulc(
    source_raster_path: Path,
    lulc_src,
) -> np.ndarray:

    """
    Reproject and align a continuous raster
    exactly to the LULC raster grid.

    Destination will have exactly the same:
    - CRS
    - transform
    - width
    - height
    - pixel grid

    Bilinear resampling is used because:
    - slope is continuous
    - elevation is continuous
    """

    with rasterio.open(
        source_raster_path
    ) as src:

        if src.crs is None:
            raise ValueError(
                f"{source_raster_path} "
                f"has no CRS."
            )

        print(
            f"\nAligning raster:"
            f"\n{source_raster_path}"
        )

        print(
            f"Source CRS: "
            f"{src.crs}"
        )

        print(
            f"Source resolution: "
            f"{src.res}"
        )

        print(
            f"Target CRS: "
            f"{lulc_src.crs}"
        )

        print(
            f"Target resolution: "
            f"{lulc_src.res}"
        )

        # ----------------------------------------------------
        # Destination array
        # ----------------------------------------------------

        destination = np.full(
            (
                lulc_src.height,
                lulc_src.width,
            ),
            FLOAT_NODATA,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Reproject + align
        # ----------------------------------------------------

        reproject(
            source=rasterio.band(
                src,
                1,
            ),

            destination=destination,

            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,

            dst_transform=lulc_src.transform,
            dst_crs=lulc_src.crs,
            dst_nodata=FLOAT_NODATA,

            # ================================================
            # CONTINUOUS DATA
            # ================================================
            #
            # Appropriate for:
            # - elevation
            # - slope
            #
            # Do not use nearest unless you specifically
            # want nearest source-pixel assignment.

            resampling=Resampling.bilinear,
        )

        return destination


# ============================================================
# EXTRACT PIXEL-LEVEL VALUES
# ============================================================

def extract_pixel_values(
    lulc_path: Path,
    slope_path: Path,
    elevation_path: Path,
) -> tuple[pd.DataFrame, dict]:

    """
    Extract slope and elevation values for every
    Grassland and Dry steppe pixel.

    Returns:
    1. Pixel-level DataFrame
    2. Dictionary containing aligned arrays
    """

    # --------------------------------------------------------
    # Open LULC raster
    # --------------------------------------------------------

    with rasterio.open(
        lulc_path
    ) as lulc_src:

        if lulc_src.crs is None:
            raise ValueError(
                "LULC raster has no CRS."
            )

        print("\nTarget LULC Raster")
        print("=" * 70)

        print(
            f"CRS: {lulc_src.crs}"
        )

        print(
            f"Resolution: {lulc_src.res}"
        )

        print(
            f"Width: {lulc_src.width}"
        )

        print(
            f"Height: {lulc_src.height}"
        )

        print(
            f"NoData: {lulc_src.nodata}"
        )

        # ----------------------------------------------------
        # Read LULC
        # ----------------------------------------------------

        lulc = lulc_src.read(1)

        # ----------------------------------------------------
        # Valid LULC pixels
        # ----------------------------------------------------

        lulc_valid = get_valid_mask(
            lulc,
            lulc_src.nodata,
        )

        # ----------------------------------------------------
        # Select only:
        #
        # 2 = Grassland
        # 3 = Dry steppe
        # ----------------------------------------------------

        target_mask = (
            lulc_valid
            & np.isin(
                lulc,
                list(
                    TARGET_CLASSES.keys()
                ),
            )
        )

        target_pixel_count = int(
            target_mask.sum()
        )

        print("\nTarget LULC Selection")
        print("=" * 70)

        print(
            f"Grassland pixels: "
            f"{np.sum(lulc == 2):,}"
        )

        print(
            f"Dry steppe pixels: "
            f"{np.sum(lulc == 3):,}"
        )

        print(
            f"Total selected pixels: "
            f"{target_pixel_count:,}"
        )

        if target_pixel_count == 0:
            raise ValueError(
                "No Grassland or Dry steppe "
                "pixels found."
            )

        # ====================================================
        # ALIGN SLOPE TO LULC GRID
        # ====================================================

        slope = (
            align_continuous_raster_to_lulc(
                source_raster_path=slope_path,
                lulc_src=lulc_src,
            )
        )

        # ====================================================
        # ALIGN ELEVATION TO LULC GRID
        # ====================================================

        elevation = (
            align_continuous_raster_to_lulc(
                source_raster_path=elevation_path,
                lulc_src=lulc_src,
            )
        )

        # ----------------------------------------------------
        # Valid slope pixels
        # ----------------------------------------------------

        slope_valid = (
            np.isfinite(slope)
            & (
                slope != FLOAT_NODATA
            )
        )

        # ----------------------------------------------------
        # Valid elevation pixels
        # ----------------------------------------------------

        elevation_valid = (
            np.isfinite(elevation)
            & (
                elevation != FLOAT_NODATA
            )
        )

        # ----------------------------------------------------
        # Final valid analysis mask
        # ----------------------------------------------------

        final_mask = (
            target_mask
            & slope_valid
            & elevation_valid
        )

        final_pixel_count = int(
            final_mask.sum()
        )

        print("\nFinal Valid Pixel Selection")
        print("=" * 70)

        print(
            f"Target LULC pixels: "
            f"{target_pixel_count:,}"
        )

        print(
            f"Pixels with valid slope "
            f"and elevation: "
            f"{final_pixel_count:,}"
        )

        print(
            f"Excluded because terrain "
            f"data missing: "
            f"{target_pixel_count - final_pixel_count:,}"
        )

        if final_pixel_count == 0:
            raise ValueError(
                "No overlapping valid pixels "
                "between LULC, slope and elevation."
            )

        # ====================================================
        # GET ROW / COLUMN INDICES
        # ====================================================

        rows, cols = np.where(
            final_mask
        )

        # ====================================================
        # GET PIXEL CENTER COORDINATES
        # ====================================================

        xs, ys = rasterio.transform.xy(
            lulc_src.transform,
            rows,
            cols,
            offset="center",
        )

        xs = np.asarray(
            xs,
            dtype=np.float64,
        )

        ys = np.asarray(
            ys,
            dtype=np.float64,
        )

        # ====================================================
        # EXTRACT VALUES
        # ====================================================

        class_values = lulc[
            final_mask
        ].astype(
            np.int16
        )

        slope_values = slope[
            final_mask
        ].astype(
            np.float32
        )

        elevation_values = elevation[
            final_mask
        ].astype(
            np.float32
        )

        # ====================================================
        # BUILD PIXEL DATAFRAME
        # ====================================================

        pixel_df = pd.DataFrame(
            {
                "row": rows,
                "col": cols,
                "x": xs,
                "y": ys,
                "lulc_value": class_values,
                "slope": slope_values,
                "elevation_m": elevation_values,
            }
        )

        # ----------------------------------------------------
        # Add class names
        # ----------------------------------------------------

        pixel_df["class_name"] = (
            pixel_df["lulc_value"]
            .map(TARGET_CLASSES)
        )

        # ----------------------------------------------------
        # Reorder columns
        # ----------------------------------------------------

        pixel_df = pixel_df[
            [
                "row",
                "col",
                "x",
                "y",
                "lulc_value",
                "class_name",
                "slope",
                "elevation_m",
            ]
        ]

        # ----------------------------------------------------
        # Store arrays for raster output
        # ----------------------------------------------------

        arrays = {
            "lulc": lulc,
            "slope": slope,
            "elevation": elevation,
            "target_mask": target_mask,
            "final_mask": final_mask,
            "profile": lulc_src.profile.copy(),
        }

        return pixel_df, arrays


# ============================================================
# CALCULATE SUMMARY STATISTICS
# ============================================================

def calculate_summary_stats(
    pixel_df: pd.DataFrame,
) -> pd.DataFrame:

    """
    Calculate slope and elevation summary
    statistics separately for:

    - Grassland
    - Dry steppe
    """

    summary = (
        pixel_df
        .groupby(
            [
                "lulc_value",
                "class_name",
            ],
            as_index=False,
        )
        .agg(
            pixel_count=(
                "lulc_value",
                "size",
            ),

            slope_min=(
                "slope",
                "min",
            ),

            slope_max=(
                "slope",
                "max",
            ),

            slope_mean=(
                "slope",
                "mean",
            ),

            slope_median=(
                "slope",
                "median",
            ),

            slope_std=(
                "slope",
                "std",
            ),

            elevation_min_m=(
                "elevation_m",
                "min",
            ),

            elevation_max_m=(
                "elevation_m",
                "max",
            ),

            elevation_mean_m=(
                "elevation_m",
                "mean",
            ),

            elevation_median_m=(
                "elevation_m",
                "median",
            ),

            elevation_std_m=(
                "elevation_m",
                "std",
            ),
        )
    )

    return summary


# ============================================================
# SAVE OUTPUT ANALYSIS RASTER
# ============================================================

def save_analysis_raster(
    arrays: dict,
    output_path: Path,
):

    """
    Save a 3-band raster:

    Band 1 = LULC class
    Band 2 = Slope
    Band 3 = Elevation

    Only Grassland and Dry steppe pixels with
    valid terrain data are retained.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lulc = arrays["lulc"]
    slope = arrays["slope"]
    elevation = arrays["elevation"]
    final_mask = arrays["final_mask"]

    profile = arrays[
        "profile"
    ].copy()

    # --------------------------------------------------------
    # Output arrays
    # --------------------------------------------------------

    lulc_output = np.full(
        lulc.shape,
        FLOAT_NODATA,
        dtype=np.float32,
    )

    slope_output = np.full(
        slope.shape,
        FLOAT_NODATA,
        dtype=np.float32,
    )

    elevation_output = np.full(
        elevation.shape,
        FLOAT_NODATA,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Keep only valid target pixels
    # --------------------------------------------------------

    lulc_output[
        final_mask
    ] = lulc[
        final_mask
    ]

    slope_output[
        final_mask
    ] = slope[
        final_mask
    ]

    elevation_output[
        final_mask
    ] = elevation[
        final_mask
    ]

    # --------------------------------------------------------
    # Update raster profile
    # --------------------------------------------------------

    profile.update(
        {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 3,
            "nodata": FLOAT_NODATA,
            "compress": "lzw",
        }
    )

    # --------------------------------------------------------
    # Save raster
    # --------------------------------------------------------

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            lulc_output,
            1,
        )

        dst.write(
            slope_output,
            2,
        )

        dst.write(
            elevation_output,
            3,
        )

        # --------------------------------
        # Band descriptions
        # --------------------------------

        dst.set_band_description(
            1,
            "LULC class",
        )

        dst.set_band_description(
            2,
            "Slope",
        )

        dst.set_band_description(
            3,
            "Elevation metres",
        )

    print("\nAnalysis raster saved to:")
    print(output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    OUTPUT_PIXEL_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_SUMMARY_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_RASTER.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # PRINT INPUT INFORMATION
    # ========================================================

    print_raster_info(
        LULC_PATH,
        "LULC Raster Information",
    )

    print_raster_info(
        SLOPE_PATH,
        "Slope Raster Information",
    )

    print_raster_info(
        ELEVATION_PATH,
        "Elevation Raster Information",
    )

    # ========================================================
    # EXTRACT PIXEL VALUES
    # ========================================================

    pixel_df, arrays = (
        extract_pixel_values(
            lulc_path=LULC_PATH,
            slope_path=SLOPE_PATH,
            elevation_path=ELEVATION_PATH,
        )
    )

    # ========================================================
    # SAVE PIXEL-LEVEL CSV
    # ========================================================

    pixel_df.to_csv(
        OUTPUT_PIXEL_CSV,
        index=False,
    )

    print("\nPixel-level CSV saved to:")
    print(OUTPUT_PIXEL_CSV)

    print(
        f"\nPixel records exported: "
        f"{len(pixel_df):,}"
    )

    # ========================================================
    # CALCULATE SUMMARY
    # ========================================================

    summary_df = (
        calculate_summary_stats(
            pixel_df
        )
    )

    # --------------------------------------------------------
    # Round summary
    # --------------------------------------------------------

    numeric_columns = (
        summary_df
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .columns
    )

    summary_df[
        numeric_columns
    ] = (
        summary_df[
            numeric_columns
        ]
        .round(2)
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\nTerrain Summary Statistics")
    print("=" * 140)

    print(
        summary_df.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE SUMMARY CSV
    # ========================================================

    summary_df.to_csv(
        OUTPUT_SUMMARY_CSV,
        index=False,
    )

    print("\nSummary CSV saved to:")
    print(OUTPUT_SUMMARY_CSV)

    # ========================================================
    # SAVE ANALYSIS RASTER
    # ========================================================

    save_analysis_raster(
        arrays=arrays,
        output_path=OUTPUT_RASTER,
    )

    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print("\nProcessing Complete")
    print("=" * 70)

    print(
        f"Selected classes: "
        f"{TARGET_CLASSES}"
    )

    print(
        f"Total valid pixel records: "
        f"{len(pixel_df):,}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()