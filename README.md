# eMOTIONAL walks, Spatial prediction of pedestrian stress in Copenhagen

Codebase for my MSc thesis at DTU Management Engineering:

**“Spatial Prediction of Pedestrian Stress Levels in Copenhagen Using Environmental Features and Machine Learning”**

The project investigates whether **micro-scale urban environmental exposures during walking** are associated with **short-term physiological stress responses** in Copenhagen, using the eMOTIONAL Cities (Daily Patterns) deployment. 

## What this repo does

High level pipeline:

1. **Walking mobility backbone**
   - Clean GPS points
   - Reconstruct walking trajectories
   - Buffer trajectories (micro-scale corridor)
   - Intersect with a **50 m hexagonal grid** and compute entry/exit times per cell. 

2. **Environmental feature extraction** on the grid
   - **NDVI** (Sentinel-2, monthly composites, interpolated for completeness)
   - **Noise** (municipal / national strategic noise maps, processed to cell-level indicators)
   - **Streetscape** from Mapillary images with semantic segmentation
   - **POIs** from OpenStreetMap
   - **Weather** (daily). 

3. **Stress label inference from wearables**
   - Subject-specific **Deep C-Means** deep clustering on Empatica E4 signals
   - Produces a continuous stress-like probability, with post hoc semantic alignment using EMA. 

4. **Models**
   - Comparison across temporal supports (high-frequency, aggregated, rolling)
   - Final analysis uses **spatial episodes** (continuous stay within one grid cell), stress is time-weighted within the episode, predictors are within-person standardized, regression uses duration-based weights. 

## Data availability (important)

This repository **does not include the original eMOTIONAL Cities raw data**.

For privacy reasons, some files and datasets are not uploaded, in particular:
- no full “diary” file with all activities
- only **walking GPS files** (or derived walking-only artifacts) are assumed
- wearable raw streams, EMA responses, and any sensitive participant-level tables are not published

The code is shared mainly to document the pipeline and support reproducibility for users with authorized access to the data.

## Repository structure

Main folders (as in the thesis workflow):

### `data_treatment/`
Mobility preprocessing and grid linking.
- `Route_clean_en.ipynb`  
  Clean GPS points and reconstruct walking trajectories.
- `gps_buffers.ipynb`  
  Buffer trajectories for micro-scale exposure corridors.
- `gps_grid_data.ipynb`  
  Grid creation and spatial joins.
- `gps_trip_time.py`  
  Utilities for trip timing and time parsing.
- `grid_times.ipynb`  
  Entry/exit time estimation per grid cell.
- `trajectory_grid_treatment.ipynb`  
  Final trajectory × grid tables used downstream.

### `feature_extraction/`
Environmental features aggregated on the grid.
- `NDVI.ipynb` and `NDVI_extractor.java`  
  NDVI extraction workflow (Sentinel-2, monthly).
- `Noise.ipynb` and `Noise_calculator.py`  
  Noise processing and grid-level indicators.
- `OSM_POI.py` and `POI.ipynb`  
  OSM POI collection and aggregation.
- `Mapillary_collect.py`  
  Download georeferenced street-level images.
- `Mapillary_segmentation.py`, `segmentation_ex.py`  
  Semantic segmentation pipeline and examples.
- `mapillary_treatment.ipynb`  
  Aggregation of segmentation outputs to grid features.
- `sunrise_sunset_scrape.py`  
  Helper for daylight-related temporal features.

### `deep_clustering/`
Wearable-based stress inference.
- `pickle_creation.py`  
  creates / updates pickles and intermediate artifacts for modeling.
- `Load_Pickles.ipynb`  
  Loading and sanity checks for generated artifacts.
- `stress_treatment_multiple.ipynb`  
  Stress probability processing and alignment to walking.
- `EMA_to_Stress_final_Analysis.ipynb`  
  Semantic alignment and EMA-based checks.

### `model/`
Statistical models and analysis notebooks.
- `AR_model_single.ipynb`  
  Autoregressive and different time aggregation checks.
- `HR_stress_model.ipynb`  
  Stress modeling variants regarding heart-rate.
- `delta_model.ipynb`  
  Delta / change-based representations.
- `model_multiple.ipynb`, `model_muliple_best_data.ipynb`  
  Main comparative runs and final model selection.

## Data download (public derived data)

The datasets required to run the notebooks are too large for GitHub.

Download the public derived data here:
- Google Drive (single zip): **[<data download link>](https://drive.google.com/drive/folders/1-NnK16PBg9o0pt-nOEmn8wVGzZPfQF-D?usp=share_link)**

After downloading:
1. Unzip the file
2. Place the extracted folder at the repository root as `./data/`  
   (keep the internal folder structure unchanged)

Expected result:
- `./data/...` contains the CSV/GeoJSON/TIF files used by the notebooks.

## How to run (practical)

Suggested order:

1. `data_treatment/Route_clean_en.ipynb`
2. `data_treatment/gps_buffers.ipynb`
3. `data_treatment/gps_grid_data.ipynb`
4. `data_treatment/grid_times.ipynb`
5. `feature_extraction/*` (NDVI, Noise, OSM, Mapillary, Weather helpers)
6. `deep_clustering/pickle_creation.py`
7. `deep_clustering/stress_treatment_multiple.ipynb`
8. `model/*`

### Environment
This repo mixes notebooks and scripts. A typical Python geo stack is needed:
- geopandas, shapely, pyproj
- pandas, numpy, scipy, scikit-learn
- matplotlib
- requests (OSM, Mapillary helpers)
- torch (if running segmentation locally)
- any project-specific dependencies used in the notebooks

Tip: create a conda env and install geopandas via conda-forge first, then pip the rest.

## Thesis reference and context

- Study context: eMOTIONAL Cities, Daily Patterns deployment in Copenhagen, wearable + mobility + environment fusion. 
- Main takeaway: pooled average associations are weak for baseline predictors (NDVI, noise, roads, buildings), heterogeneity across participants is strong, and the exposure definition matters a lot. 

## Notes

- Paths, tokens, API keys (Mapillary, etc.) are not included. Use environment variables or a local `.env` and keep it out of git.
- Some notebooks assume precomputed intermediate files (pickles, cached downloads). That is intentional, the pipeline is heavy.

## License  
Right now: **no license specified**.

## Citation
If you use this codebase, cite the thesis:

> Piccagnoni, M. (2026). *Spatial Prediction of Pedestrian Stress Levels in Copenhagen Using Environmental Features and Machine Learning*. MSc Thesis, DTU.