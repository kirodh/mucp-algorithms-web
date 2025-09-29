"""
Purpose: Read data functions for MUCP file data, mbal, miu, compartment, gis mapping etc, shp, excel and csv
Author: Kirodh Boodhraj
"""

import pandas as pd
import geopandas as gpd
import re
import numpy as np
from .data_validators import  validate_miu_shapefile, validate_nbal_shapefile, validate_compartment_shapefile, validate_gis_mapping_shapefile, validate_miu_linked_species_excel, validate_nbal_linked_species_excel, validate_compartment_priorities_csv

# helper functions:

# normalize the stages to numerical values
def normalize_stage(value: str) -> int:
    value = str(value).strip().lower()
    if value == "initial treatment":
        return 0
    elif value == "maintenance":
        return -1
    else:
        # Extract number from string like "5th follow up"
        match = re.search(r"(\d+)", value)
        if match:
            return int(match.group(1))
        else:
            # If no number found, return NaN or raise error depending on your needs
            return None



# Main reader and cleaner functions

# MIU shp
def read_miu_shapefile(path: str,
                       gis_mapping_miu_ids: list,
                       validate: bool = False,
                       headers_required: list = ["miu_id", "area", "riparian_c"],
                       headers_other: list = ["geometry"],
                       ) -> gpd.GeoDataFrame | dict:
    """
    Read and optionally validate or prepare a MIU shapefile.

    Args:
        path (str): Path to the shapefile.
        validate (bool): If True, only run validation and return warnings/errors.
                         If False, load and clean the data.

    Returns:
        dict: If validate=True, returns {"warnings": [...], "errors": [...]}.
        GeoDataFrame: If validate=False, returns cleaned GeoDataFrame.
        :param headers_required:
        :param headers_other:
    """
    if validate:
        return validate_miu_shapefile(path, gis_mapping_miu_ids, headers_required, headers_other)

    # --- PREPARATION / CLEANING ---
    gdf = gpd.read_file(path)
    gdf.columns = [c.lower() for c in gdf.columns]

    # no checks because assume validation was done

    # clean data operations

    miu_id_header = headers_required[0]
    area_header = headers_required[1]
    riparian_header = headers_required[2]
    geometry_header = headers_other[0]

    #1.  get only entries that are linked in the gis mapping file, discard all other entries:
    # Get all unique non-null values from the gdf column
    unique_ids = set(gdf[miu_id_header].dropna().str.lower().unique())

    # Ensure gis_mapping_miu_ids is unique too
    gis_mapping_miu_ids_set = set(gis_mapping_miu_ids)

    # Find which IDs are missing
    missing_ids = gis_mapping_miu_ids_set - unique_ids

    # Drop rows whose miu_id_header is in missing_ids
    gdf = gdf[~gdf[miu_id_header].str.lower().isin(missing_ids)]


    #2. Drop rows with missing critical fields
    for field in headers_required:
        gdf = gdf[gdf[field].notna()]

    # 3. Normalize riparian to only 'l' or 'r'
    gdf[riparian_header] = gdf[riparian_header].astype(str).str.lower().str.strip()

    #4. Merge duplicates by MIU id, sum areas, prefer riparian='r'
    gdf = (
        gdf.sort_values(by=[riparian_header], ascending=True)
            .groupby(miu_id_header, as_index=False)
            .agg({
            area_header: "sum",
            riparian_header: "last",
            geometry_header: "first"
        })
    )

    # Strip strings in object fields
    for col in gdf.select_dtypes(include=["object"]).columns:
        gdf[col] = gdf[col].str.strip().str.lower()

    return gdf


# NBAL shp
def read_nbal_shapefile(path: str,
                       gis_mapping_nbal_ids: list,
                       validate: bool = False,
                       headers_required: list = ["nbal_id", "area", "stage"],
                       headers_other: list = ["geometry", "contractid", "first_date", "last_date"],
                       ) -> gpd.GeoDataFrame | dict:
    """
    Read and optionally validate or prepare a MIU shapefile.

    Args:
        path (str): Path to the shapefile.
        validate (bool): If True, only run validation and return warnings/errors.
                         If False, load and clean the data.

    Returns:
        dict: If validate=True, returns {"warnings": [...], "errors": [...]}.
        GeoDataFrame: If validate=False, returns cleaned GeoDataFrame.
        :param headers_required:
        :param headers_other:
    """
    if validate:
        return validate_nbal_shapefile(path, gis_mapping_nbal_ids, headers_required, headers_other)

    # --- PREPARATION / CLEANING ---
    gdf = gpd.read_file(path)
    gdf.columns = [c.lower() for c in gdf.columns]

    # no checks because assume validation was done

    # clean data operations

    nbal_id_header = headers_required[0]
    area_header = headers_required[1]
    stage_header = headers_required[2]
    geometry_header = headers_other[0]

    # 1.  get only entries that are linked in the gis mapping file, discard all other entries:
    # Get all unique non-null values from the gdf column
    unique_ids = set(gdf[nbal_id_header].dropna().str.lower().unique())

    # Ensure gis_mapping_nbal_ids is unique too
    gis_mapping_nbal_ids_set = set(gis_mapping_nbal_ids)

    # Find which IDs are missing
    missing_ids = unique_ids - gis_mapping_nbal_ids_set

    # Drop rows whose miu_id_header is in missing_ids
    gdf = gdf[~gdf[nbal_id_header].str.lower().isin(missing_ids)]

    #2. Drop rows with missing critical fields
    for field in headers_required:
        gdf = gdf[gdf[field].notna()]


    #3. Merge duplicates by NBAL id, sum areas, prefer just a number for the stage
    gdf = (
        gdf.sort_values(by=[nbal_id_header], ascending=True)
            .groupby(nbal_id_header, as_index=False)
            .agg({
            area_header: "sum",
            geometry_header: "first",
            stage_header: "first"  # keep one stage for now, will normalize next
        })
    )

    if stage_header in gdf.columns:
        gdf[stage_header] = gdf[stage_header].apply(normalize_stage)

    #4. Strip strings in object fields
    for col in gdf.select_dtypes(include=["object"]).columns:
        gdf[col] = gdf[col].str.strip().str.lower()

    return gdf


# compartment shp
def read_compartment_shapefile(path: str,
                       gis_mapping_compartment_ids: list,
                       validate: bool = False,
                       headers_required: list = ["compt_id", "area", "slope","walk_time","drive_time","costing","grow_con"],
                       headers_other: list = ["geometry", "terrain"],
                       ) -> gpd.GeoDataFrame | dict:
    """
    Read and optionally validate or prepare a MIU shapefile.

    Args:
        path (str): Path to the shapefile.
        validate (bool): If True, only run validation and return warnings/errors.
                         If False, load and clean the data.

    Returns:
        dict: If validate=True, returns {"warnings": [...], "errors": [...]}.
        GeoDataFrame: If validate=False, returns cleaned GeoDataFrame.
        :param headers_required:
        :param headers_other:
    """
    if validate:
        return validate_compartment_shapefile(path, gis_mapping_compartment_ids, headers_required, headers_other)

    # --- PREPARATION / CLEANING ---
    gdf = gpd.read_file(path)
    gdf.columns = [c.lower() for c in gdf.columns]

    # no checks because assume validation was done

    # clean data operations

    compartment_id_header = headers_required[0]
    area_header = headers_required[1]
    slope_header = headers_required[2]
    walk_time_header = headers_required[3]
    drive_time_header = headers_required[4]
    costing_model_header = headers_required[5]
    growth_condition_header = headers_required[6]
    geometry_header = headers_other[0]

    # 1.  get only entries that are linked in the gis mapping file, discard all other entries:
    # Get all unique non-null values from the gdf column
    unique_ids = set(gdf[compartment_id_header].dropna().str.lower().unique())

    # Ensure gis_mapping_compartment_ids is unique too
    gis_mapping_compartment_ids_set = set(gis_mapping_compartment_ids)

    # Find which IDs are missing
    missing_ids = unique_ids - gis_mapping_compartment_ids_set

    # Drop rows whose miu_id_header is in missing_ids
    gdf = gdf[~gdf[compartment_id_header].str.lower().isin(missing_ids)]
    #2. Drop rows with missing critical fields
    for field in headers_required:
        gdf = gdf[gdf[field].notna()]

    #3. --- STEP: Merge duplicates by Compartment ID ---
    gdf = (
        gdf.sort_values(by=[compartment_id_header], ascending=True)
            .groupby(compartment_id_header, as_index=False)
            .agg({
            area_header: "sum",  # sum all areas
            slope_header: "first",  # keep first slope
            walk_time_header: "first",  # keep first walk time
            drive_time_header: "first",  # keep first drive time
            costing_model_header: "first",  # keep first costing model
            growth_condition_header: "first",  # keep first growth condition
            geometry_header: "first"  # keep first geometry
        })
    )

    #4. Strip strings in object fields
    for col in gdf.select_dtypes(include=["object"]).columns:
        gdf[col] = gdf[col].str.strip().str.lower()

    return gdf


# GIS mapping shp
def read_gis_mapping_shapefile(path: str,
                       validate: bool = False,
                       headers_required: list = ["nbal_id", "miu_id", "compt_id","area"],
                       headers_other: list = ["geometry"],
                       ) -> gpd.GeoDataFrame | dict:
    """
    Read and optionally validate or prepare a MIU shapefile.

    Args:
        path (str): Path to the shapefile.
        validate (bool): If True, only run validation and return warnings/errors.
                         If False, load and clean the data.

    Returns:
        dict: If validate=True, returns {"warnings": [...], "errors": [...]}.
        GeoDataFrame: If validate=False, returns cleaned GeoDataFrame.
        :param headers_required:
        :param headers_other:
    """
    if validate:
        return validate_gis_mapping_shapefile(path, headers_required, headers_other)

    # --- PREPARATION / CLEANING ---
    gdf = gpd.read_file(path)
    gdf.columns = [c.lower() for c in gdf.columns]

    # no checks because assume validation was done

    # clean data operations

    nbal_id_header = headers_required[0]
    miu_id_header = headers_required[1]
    compartment_id_header = headers_required[2]
    area_header = headers_required[3]
    geometry_header = headers_other[0]

    # Drop rows with missing critical fields
    for field in headers_required:
        gdf = gdf[gdf[field].notna()]

    # Fill empty IDs with None (instead of NaN/empty string for grouping consistency)
    gdf[nbal_id_header] = gdf[nbal_id_header].replace({np.nan: None, "": None})
    gdf[miu_id_header] = gdf[miu_id_header].replace({np.nan: None, "": None})
    gdf[compartment_id_header] = gdf[compartment_id_header].replace({np.nan: None, "": None})

    # Create a grouping key based on the rules
    def make_group_key(row):
        if row[nbal_id_header] is not None:  # nbal present → strictest
            return ("nbal", row[nbal_id_header], row[miu_id_header], row[compartment_id_header])
        elif row[miu_id_header] is not None:  # miu present → miu+compartment
            return ("miu", row[miu_id_header], row[compartment_id_header])
        else:  # only compartment
            return ("compartment", row[compartment_id_header])

    gdf["_merge_key"] = gdf.apply(make_group_key, axis=1)

    # Aggregate by merge key
    gdf = (
        gdf.sort_values(by=[compartment_id_header], ascending=True)
            .groupby("_merge_key", as_index=False)
            .agg({
            nbal_id_header: "first",  # keep first filled
            miu_id_header: "first",
            compartment_id_header: "first",
            area_header: "sum",  # sum all areas
            geometry_header: "first"  # keep first geometry
        })
    )

    # Drop the helper key
    gdf = gdf.drop(columns=["_merge_key"])

    # Strip strings in object fields
    for col in gdf.select_dtypes(include=["object"]).columns:
        gdf[col] = gdf[col].str.strip().str.lower()

    # last step remove by Identify compartments that are linked to nbal or miu
    compartments_with_links = set(
        gdf.loc[
            gdf[nbal_id_header].notna() | gdf[miu_id_header].notna(),
            compartment_id_header
        ]
    )

    # Drop "compartment-only" rows if their compartment also exists in linked rows
    gdf = gdf[
        ~(
                gdf[nbal_id_header].isna() &
                gdf[miu_id_header].isna() &
                gdf[compartment_id_header].isin(compartments_with_links)
        )
    ]

    return gdf


# MIU linked species Excel
def read_miu_linked_species_excel(path: str,
                                  validate: bool = False,
                                  headers_required: list = ["miu_id", "species", "idenscode", "age"],) -> pd.DataFrame:
    if validate:
        return validate_miu_linked_species_excel(path, headers_required)

        # --- PREPARATION / CLEANING ---
    df = pd.read_excel(path)
    df.columns = [c.lower() for c in df.columns]

    # no checks because assume validation was done

    # clean data operations

    miu_id_header = headers_required[0]
    species_header = headers_required[1]
    idenscode_header = headers_required[2]
    age_header = headers_required[3]

    # Drop rows with missing critical fields
    for field in headers_required:
        df = df[df[field].notna()]

    # Fill empty IDs with None (instead of NaN/empty string for grouping consistency)
    df[miu_id_header] = df[miu_id_header].replace({np.nan: None, "": None})
    df[species_header] = df[species_header].replace({np.nan: None, "": None})
    df[idenscode_header] = df[idenscode_header].replace({np.nan: None, "": None})
    df[age_header] = df[age_header].replace({np.nan: None, "": None})

    # Drop duplicate rows based on all four columns
    df = df.drop_duplicates(subset=[miu_id_header, species_header, idenscode_header, age_header], keep="first").reset_index(drop=True)

    # Strip strings in object fields
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip().str.lower()

    return df


# NBAL linked species Excel
def read_nbal_linked_species_excel(path: str,
                                   validate: bool = False,
                                   headers_required: list = ["nbal_id", "miu_id", "compt_id","area"],) -> pd.DataFrame:
    if validate:
        return validate_nbal_linked_species_excel(path, headers_required)

        # --- PREPARATION / CLEANING ---
    df = pd.read_excel(path)
    df.columns = [c.lower() for c in df.columns]

    # no checks because assume validation was done

    # clean data operations

    nbal_id_header = headers_required[0]
    species_header = headers_required[1]
    idenscode_header = headers_required[2]
    age_header = headers_required[3]

    # Drop rows with missing critical fields
    for field in headers_required:
        df = df[df[field].notna()]

    # Fill empty IDs with None (instead of NaN/empty string for grouping consistency)
    df[nbal_id_header] = df[nbal_id_header].replace({np.nan: None, "": None})
    df[species_header] = df[species_header].replace({np.nan: None, "": None})
    df[idenscode_header] = df[idenscode_header].replace({np.nan: None, "": None})
    df[age_header] = df[age_header].replace({np.nan: None, "": None})

    # Drop duplicate rows based on all four columns
    df = df.drop_duplicates(subset=[nbal_id_header, species_header, idenscode_header, age_header],
                            keep="first").reset_index(drop=True)

    # Strip strings in object fields
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip().str.lower()

    return df


# compartment priorities CSV
def read_compartment_priorities_csv(path: str, validate: bool = False, headers_required: list = ["compt_id"]) -> pd.DataFrame:
    if validate:
        return validate_compartment_priorities_csv(path, headers_required)

        # --- PREPARATION / CLEANING ---
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    # no checks because assume validation was done

    # clean data operations

    compt_id_header = headers_required[0]

    # Drop rows with missing critical fields
    df = df.dropna()


    # Drop duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Strip strings in object fields
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip().str.lower()

    return df
