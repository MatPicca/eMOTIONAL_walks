import geopandas as gpd
import os 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from shapely.geometry import Point
from tqdm import tqdm
from scipy.stats import norm, CensoredData # to treat censored data under detection limits


# function to assign fixed 5 dB bands based on isov1/isov2 or Lden
def _assign_fixed_band(row):
    BINS = [(45,50),(50,55),(55,60),(60,65),(65,70),(70,75),(75,80)]
    if pd.notnull(row.get("isov1", np.nan)) and pd.notnull(row.get("isov2", np.nan)):
        lo, hi = float(row["isov1"]), float(row["isov2"])
    else:
        L = float(row["Lden"])
        lo = 5 * np.floor(L / 5.0)
        hi = lo + 5

    # snap lower and upper bounds to nearest multiple of 5 within [45,80] for column consistency
    lo_fixed = min(max(45, 5 * round(lo / 5)), 75)
    hi_fixed = min(lo_fixed + 5, 80)
    return f"{int(lo_fixed)}_{int(hi_fixed)}"

# --- Area-weighted arithmetic noise calculation ---
def area_weighted_noise_for_cell_arith(hex_geom, src_layers, quiet_value):
    BINS = [(45,50),(50,55),(55,60),(60,65),(65,70),(70,75),(75,80)]
    cell_area = hex_geom.area
    all_segments = []

    for src_name, src_gdf in src_layers.items():
        cols = ["Lden", "geometry"]
        if "isov1" in src_gdf.columns and "isov2" in src_gdf.columns:
            cols = ["Lden", "isov1", "isov2", "geometry"]
        inter = gpd.overlay(
            gpd.GeoDataFrame(geometry=[hex_geom], crs=src_gdf.crs),
            src_gdf[cols],
            how="intersection"
        )
        if inter.empty:
            continue
        inter["area"] = inter.geometry.area
        inter["source"] = src_name
        all_segments.append(inter)

    # No polygons at all: cell is quiet
    if len(all_segments) == 0:
    # assume L ~ Uniform[L_low, quiet_value]
        L_mean = 0.5 * ((quiet_value - 5) + quiet_value)
        L_std = (quiet_value - (quiet_value - 5)) / np.sqrt(12)

        out = {
            "L_mean": L_mean,
            "L_std": L_std,
            "L_min": quiet_value,
            "L_max": quiet_value,
            "share_quiet_": 1.0,
        }
        for bl, bh in BINS:
            out[f"share_{bl}_{bh}"] = 0.0
        return out

    inter_all = pd.concat(all_segments, ignore_index=True)

    # Raw coverage weights (can exceed 1 with overlaps)
    inter_all["w"] = inter_all.geometry.area / cell_area
    # Assign fixed 5 dB band labels
    inter_all["band"] = inter_all.apply(_assign_fixed_band, axis=1)

    # Add quiet for uncovered fraction (with tolerance)
    covered_fraction = inter_all["w"].sum()
    tolerance = 1e-3  # 0.1%
    quiet_fraction = 1.0 - covered_fraction
    if quiet_fraction >= tolerance:
        inter_all = pd.concat([inter_all, pd.DataFrame({
            "Lden": [quiet_value],
            "w": [quiet_fraction],
            "band": ["quiet_"]
        })], ignore_index=True)

    # Normalize weights so they sum to 1 (this is the key change)
    total_w = inter_all["w"].sum()
    if total_w <= 0:
        inter_all["w_norm"] = 0.0
    else:
        inter_all["w_norm"] = inter_all["w"] / total_w # normalized fraction of noise coverage

    # Arithmetic, coverage-normalized stats in dB
    # L_mean = float((inter_all["Lden"] * inter_all["w_norm"]).sum())
    # var = float((inter_all["w_norm"] * (inter_all["Lden"] - L_mean) ** 2).sum())
    # L_std = float(np.sqrt(max(var, 0.0)))
    # For min/max, ignore tiny quiet slivers by the same tolerance on w_norm

    # Arithmetic stats in dB with censored treatment 
    CENSOR_LIMIT = quiet_value  

    L_vals = inter_all["Lden"].to_numpy()
    censored_mask = L_vals <= CENSOR_LIMIT 
    # print(censored_mask.sum())

    try:
        data = CensoredData.left_censored(L_vals, censored_mask)
        loc, scale = norm.fit(data)
        L_mean = float(loc)
        L_std = float(scale)
    except Exception:
        # fallback se tutti sono censurati o il fit non converge
        if np.all(censored_mask):
            L_mean = CENSOR_LIMIT - 5.0
            L_std = 0.0
        else:
            L_mean = float((inter_all["Lden"] * inter_all["w_norm"]).sum())
            var = float((inter_all["w_norm"] * (inter_all["Lden"] - L_mean) ** 2).sum())
            L_std = float(np.sqrt(max(var, 0.0)))


    small = inter_all["w_norm"] >= tolerance
    L_min = float(inter_all.loc[small, "Lden"].min()) if small.any() else float(inter_all["Lden"].min())
    L_max = float(inter_all.loc[small, "Lden"].max()) if small.any() else float(inter_all["Lden"].max())

    # Per-band “shares” based on normalized coverage (overlap-aware; sum to 1)
    band_shares = (
        inter_all.groupby("band", as_index=False)["w_norm"]
                 .sum()
                 .rename(columns={"w_norm": "share"})
    )

    # Build full output with fixed bins always present
    out = {"L_mean": L_mean, "L_std": L_std, "L_min": L_min, "L_max": L_max}
    for bl, bh in BINS: out[f"share_{bl}_{bh}"] = 0.0
    out["share_quiet_"] = 0.0
    for _, r in band_shares.iterrows():
        out[f"share_{r['band']}"] = float(r["share"])

    # Tiny numerical closure note (optional)
    total_share = sum(v for k, v in out.items() if k.startswith("share_"))
    if abs(total_share - 1.0) > 1e-6:
        out["share_adjust_note"] = total_share

    return out

# --- Area-weighted energetic noise calculation ---
def area_weighted_noise_for_cell_energetic(hex_geom, src_layers, quiet_value):
    BINS = [(45,50),(50,55),(55,60),(60,65),(65,70),(70,75),(75,80)]
    """
    Combine multiple sources (road, rail, airport) via area intersection.
    Returns energetic stats + per-band disjoint coverage (shares 0–1)
    based on fixed bins 45–80 dB (5 dB width).
    """
    cell_area = hex_geom.area
    all_segments = []

    for src_name, src_gdf in src_layers.items():
        cols = ["Lden", "geometry"]
        if "isov1" in src_gdf.columns and "isov2" in src_gdf.columns:
            cols = ["Lden", "isov1", "isov2", "geometry"]
        inter = gpd.overlay(
            gpd.GeoDataFrame(geometry=[hex_geom], crs=src_gdf.crs),
            src_gdf[cols],
            how="intersection"
        )
        if inter.empty:
            continue
        inter["area"] = inter.geometry.area
        inter["source"] = src_name
        all_segments.append(inter)

    if len(all_segments) == 0:
        # Entire cell is quiet → only quiet band
        out = {
            "L_mean": quiet_value,
            "L_std": 0.0,
            "L_min": quiet_value,
            "L_max": quiet_value,
            "share_quiet_": 1.0,
        }
        for bl, bh in BINS:
            out[f"share_{bl}_{bh}"] = 0.0
        return out

    # Combine all intersections
    inter_all = pd.concat(all_segments, ignore_index=True)
    inter_all["w"] = inter_all.geometry.area / cell_area

    # --- assign bands right away ---
    inter_all["band"] = inter_all.apply(_assign_fixed_band, axis=1)

    # --- handle coverage ---
    covered_fraction = inter_all["w"].sum()
    tolerance = 1e-3  # 0.1% of cell area
    quiet_fraction = 1.0 - covered_fraction
    if quiet_fraction < tolerance:
        quiet_fraction = 0.0  # ignore negligible uncovered parts

    # Add quiet band only if meaningful uncovered area
    if quiet_fraction > 0:
        inter_all = pd.concat([
            inter_all,
            pd.DataFrame({
                "Lden": [quiet_value],
                "w": [quiet_fraction],
                "band": [f"quiet_"]
            })
        ], ignore_index=True)

    # Energetic combination
    energy_sum = np.sum(inter_all["w"] * 10.0**(inter_all["Lden"]/10.0))
    L_mean = 10.0 * np.log10(max(energy_sum, 1e-12))
    var = np.sum(inter_all["w"] * (inter_all["Lden"] - L_mean)**2)
    L_std = np.sqrt(max(var, 0.0))
    L_min = inter_all["Lden"].min()
    L_max = inter_all["Lden"].max()

    # Shares per fixed band
    band_shares = (
        inter_all.groupby("band", as_index=False)["w"]
                 .sum()
                 .rename(columns={"w": "share"})
    )

    # Build full output
    out = {
        "L_mean": L_mean,
        "L_std": L_std,
        "L_min": L_min,
        "L_max": L_max,
    }

    # Initialize all bins with 0
    for bl, bh in BINS:
        out[f"share_{bl}_{bh}"] = 0.0
    out["share_quiet_"] = 0.0

    # Fill actual shares
    for _, r in band_shares.iterrows():
        out[f"share_{r['band']}"] = float(r["share"])

    # Ensure closure
    total_share = sum(v for k, v in out.items() if k.startswith("share_"))
    if abs(total_share - 1.0) > 1e-3:
        out["share_adjust_note"] = total_share

    out["is_censored"] = 1 if out.get("share_quiet_", 0) > 0 else 0

    return out





# paths
base_path = "/home/s232713/data/Noise/filtered"
grid_json = "/home/s232713/data/grid_data/cph_hexgrid.geojson" 
active_csv = "/home/s232713/data/grid_data/grid_to_trip.csv"

# active hexagons
active = pd.read_csv(active_csv)
active_ids = set(active["GRID_ID"].unique())

# full grid + active hexagons
grid = gpd.read_file(grid_json)
grid = grid[grid["GRID_ID"].isin(active_ids)].reset_index(drop=True)
print(f"{len(grid)} active hexagons") # (12920)

# noise
road_day = gpd.read_file(f"{base_path}/road_noise_1_5m_clipped.gpkg")
rail_day = gpd.read_file(f"{base_path}/rail_noise_1_5m_clipped.gpkg")
airport_day = gpd.read_file(f"{base_path}/airport_civil_noise_1_5m_clipped.gpkg")

road_night = gpd.read_file(f"{base_path}/road_night_noise_1_5m_clipped.gpkg")
rail_night = gpd.read_file(f"{base_path}/rail_night_noise_1_5m_clipped.gpkg")
airport_night = gpd.read_file(f"{base_path}/airport_civil_night_noise_1_5m_clipped.gpkg")

# Fix column naming for airport
for df in [airport_day, airport_night]:
    df.rename(columns={"iso1": "isov1", "iso2": "isov2"}, inplace=True)
# print('airport day columns:', airport_day.columns)
# print('airport night columns:', airport_night.columns)

noise_layers_day = {"road": road_day, "rail": rail_day, "airport": airport_day}
noise_layers_night = {"road": road_night, "rail": rail_night, "airport": airport_night}


# Ensure CRS matches grid and add Lden column (midpoint)
for name, gdf in list(noise_layers_day.items()) + list(noise_layers_night.items()):
    gdf["Lden"] = gdf[["isov1", "isov2"]].mean(axis=1)
    gdf.to_crs(grid.crs, inplace=True)

# print('airport columns:', airport.columns)


QUIET_VALUE_DAY = 55.0  # justified with Nord2000 for <55 dB roads
QUIET_VALUE_NIGHT = 45.0  # justified for <45 dB roads

records_day = []
records_night = []

for idx, row in tqdm(grid.iterrows(), total=len(grid)):
    gid = row["GRID_ID"]
    geom = row.geometry

    # DAY
    vals_day = area_weighted_noise_for_cell_arith(geom, noise_layers_day, QUIET_VALUE_DAY)
    vals_day["GRID_ID"] = gid
    records_day.append(vals_day)

    # NIGHT
    vals_night = area_weighted_noise_for_cell_arith(geom, noise_layers_night, QUIET_VALUE_NIGHT)
    vals_night["GRID_ID"] = gid
    records_night.append(vals_night)

records_day = pd.DataFrame(records_day)
records_night = pd.DataFrame(records_night)

# censored flag column 
records_day["is_censored"] = (records_day["share_quiet_"] > 0).astype(int)
records_night["is_censored"] = (records_night["share_quiet_"] > 0).astype(int)

print("Day stats sample:\n", records_day.head(3))
print("\nNight stats sample:\n", records_night.head(3))

records_day.to_csv("/home/s232713/data/Noise/grid_noise_day.csv", index=False)
records_night.to_csv("/home/s232713/data/Noise/grid_noise_night.csv", index=False)
