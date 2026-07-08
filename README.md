# Mongolia Geospatial Analysis

A Python-based geospatial analysis project for processing, analyzing, and visualizing spatial data for Mongolia.

## Project Structure

```text
mongolia-geospatial-analysis/
├── config/              # Project configuration files
├── data/
│   └── raw/             # Raw spatial data (not tracked by Git)
├── notebooks/           # Jupyter notebooks for exploration
├── outputs/
│   ├── maps/            # Generated maps
│   └── stats/           # Statistical outputs
├── scripts/             # Workflow and processing scripts
├── src/
│   ├── analysis/        # Spatial analysis modules
│   ├── data_processing/ # Data preprocessing modules
│   ├── utils/           # Utility functions
│   └── visualization/   # Mapping and visualization modules
├── environment.yml      # Conda environment
├── requirements.txt     # Python dependencies
└── README.md
```

## Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate mongolia-project
```

## Data

Large geospatial datasets are not stored in this repository.

The following file types are excluded from Git:

* Shapefiles
* GeoTIFF rasters
* GeoPackages
* Satellite imagery
* Large processed datasets
* Generated outputs

Place local input datasets inside:

```text
data/raw/
```

## Main Tools

* Python
* GeoPandas
* Rasterio
* GDAL
* Shapely
* PyProj
* Xarray
* Matplotlib
* JupyterLab

## Outputs

Generated project outputs are stored in:

```text
outputs/maps/
outputs/stats/
```
