import os
import numpy as np
import pandas as pd
import geopandas as gpd
import re


def validate_miu_shapefile(path: str, gis_mapping_miu_ids: list, headers_required: list, headers_other: list) -> dict:
    """
        Validate MIU shapefile structure and data.

        Returns:
            dict with "errors" (blocking) and "warnings" (non-blocking).
        """

    warnings, errors = [], []

    miu_id_header = headers_required[0]
    area_header = headers_required[1]
    riparian_header = headers_required[2]

    # --- STEP 1: Try opening file ---
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        return {"warnings": warnings, "errors": [f"Could not open shapefile: {e}"]}

    # --- STEP 2: Standardize headers ---
    gdf.columns = [c.lower() for c in gdf.columns]

    # --- STEP 3: Check required columns ---
    missing_columns = [c for c in headers_required if c not in gdf.columns]
    if missing_columns:
        return {"warnings": warnings, "errors": [f"Missing required columns: {missing_columns}"]}

    # --- STEP 4: Error checks ---
    # Area must be numeric
    if area_header in gdf.columns:  # area
        # Find non-numeric values
        non_float_area = gdf[~pd.to_numeric(gdf[area_header], errors="coerce").notna()]
        for idx, row in non_float_area.iterrows():
            errors.append(f"Row {idx}: area is not numeric → {row[area_header]}")

        # Convert to numeric safely
        gdf[area_header] = pd.to_numeric(gdf[area_header], errors="coerce")

        # Find negative values
        negative_area = gdf[gdf[area_header] < 0]
        for idx, row in negative_area.iterrows():
            errors.append(f"Row {idx}: area is negative → {row[area_header]}")


    # Riparian_c must be "l" or "r"
    if riparian_header in gdf.columns:
        invalid_riparian = gdf[~gdf[riparian_header].astype(str).str.lower().isin(["l", "r"])]
        for idx, row in invalid_riparian.iterrows():
            errors.append(f"Row {idx}: riparian_c invalid → {row[riparian_header]}")

    # If any errors exist → return immediately
    if errors:
        return {"warnings": warnings, "errors": errors}

    # --- STEP 5: Warning checks ---
    # 5.1 Missing optional fields
    for field in headers_other:
        if field in gdf.columns:
            missing_rows = gdf[gdf[field].isna()]
            for idx, _ in missing_rows.iterrows():
                warnings.append(f"Row {idx}: missing {field}")

    # 5.2 Duplicate MIU IDs
    if miu_id_header in gdf.columns:
        duplicates = gdf[gdf.duplicated(subset=[miu_id_header], keep=False)]
        for idx, row in duplicates.iterrows():
            warnings.append(f"Row {idx}: duplicate miu_id -> {row[miu_id_header]}")

    # 5.3 Find rows where ANY required column is empty/NaN
    for col in headers_required:
        # Find rows where this column is NaN or just whitespace
        missing_rows = gdf[gdf[col].isna() | (gdf[col].astype(str).str.strip() == "")]

        # Collect warnings per row
        for idx, row in missing_rows.iterrows():
            warnings.append(f"Row {idx}: missing value(s) in required columns → {row[headers_required].to_dict()}")

    # ---Find if the miu id's are in the gis mapping or if any missing
    # Get all unique non-null values from the gdf column
    unique_ids = set(gdf[miu_id_header].dropna().str.lower().unique())

    # Ensure gis_mapping_miu_ids is unique too
    gis_mapping_miu_ids_set = set(gis_mapping_miu_ids)

    # Find which IDs are missing
    missing_ids = gis_mapping_miu_ids_set - unique_ids

    # Append warnings for each missing ID
    for idx in missing_ids:
        warnings.append(f" GIS mapping file MIU ID {idx}: not in the MIU shapefile.")

    # --- STEP 6: Return results ---
    return {"warnings": warnings, "errors": errors}

def validate_nbal_shapefile(path: str, gis_mapping_nbal_ids: list, headers_required: list, headers_other: list) -> dict:
    """
        Validate MIU shapefile structure and data.

        Returns:
            dict with "errors" (blocking) and "warnings" (non-blocking).
        """

    warnings, errors = [], []

    nbal_id_header = headers_required[0]
    area_header = headers_required[1]
    stage_header = headers_required[2]

    # --- STEP 1: Try opening file ---
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        return {"warnings": warnings, "errors": [f"Could not open shapefile: {e}"]}

    # --- STEP 2: Standardize headers ---
    gdf.columns = [c.lower() for c in gdf.columns]

    # --- STEP 3: Check required columns ---
    missing_columns = [c for c in headers_required if c not in gdf.columns]
    if missing_columns:
        return {"warnings": warnings, "errors": [f"Missing required columns: {missing_columns}"]}

    # --- STEP 4: Error checks ---
    # Area must be numeric
    if area_header in gdf.columns:  # area
        # Find non-numeric values
        non_float_area = gdf[~pd.to_numeric(gdf[area_header], errors="coerce").notna()]
        for idx, row in non_float_area.iterrows():
            errors.append(f"Row {idx}: area is not numeric → {row[area_header]}")

        # Convert to numeric safely
        gdf[area_header] = pd.to_numeric(gdf[area_header], errors="coerce")

        # Find negative values
        negative_area = gdf[gdf[area_header] < 0]
        for idx, row in negative_area.iterrows():
            errors.append(f"Row {idx}: area is negative → {row[area_header]}")

    # check for follow up or maintenance or initial treatment
    if stage_header in gdf.columns:
        # normalize values to lowercase stripped strings
        stage_values = gdf[stage_header].astype(str).str.strip().str.lower()

        # valid fixed values
        valid_fixed = {"maintenance", "initial treatment"}

        # regex pattern for follow-up stages
        followup_pattern = re.compile(r"^\d+(st|nd|rd|th)\s+follow up$")

        # boolean mask of valid rows
        valid_mask = (
                stage_values.isin(valid_fixed)
                | stage_values.str.match(followup_pattern)
        )

        # invalid rows
        invalid_rows = gdf.loc[~valid_mask, stage_header]

        # collect error messages
        for idx, val in invalid_rows.items():
            errors.append(f"Row {idx}: invalid stage value → {val} (must be 'maintenance', 'initial treatment', or e.g. '5th follow up')")


    # If any errors exist → return immediately
    if errors:
        return {"warnings": warnings, "errors": errors}

    # --- STEP 5: Warning checks ---
    # 5.1 Missing optional fields
    for field in headers_other:
        if field in gdf.columns:
            missing_rows = gdf[gdf[field].isna()]
            for idx, _ in missing_rows.iterrows():
                warnings.append(f"Row {idx}: missing {field}")

    # 5.2 Duplicate NBAL IDs
    if nbal_id_header in gdf.columns:
        duplicates = gdf[gdf.duplicated(subset=[nbal_id_header], keep=False)]
        for idx, row in duplicates.iterrows():
            warnings.append(f"Row {idx}: duplicate nbal_id -> {row[nbal_id_header]}")


    # 5.3 Find rows where ANY required column is empty/NaN
    for col in headers_required:
        # Find rows where this column is NaN or just whitespace
        missing_rows = gdf[gdf[col].isna() | (gdf[col].astype(str).str.strip() == "")]

        # Collect warnings per row
        for idx, row in missing_rows.iterrows():
            warnings.append(f"Row {idx}: missing value(s) in required columns → {row[headers_required].to_dict()}")

    # ---Find if the miu id's are in the gis mapping or if any missing
    # Get all unique non-null values from the gdf column
    unique_ids = set(gdf[nbal_id_header].dropna().str.lower().unique())

    # Ensure gis_mapping_miu_ids is unique too
    gis_mapping_nbal_ids_set = set(gis_mapping_nbal_ids)

    # Find which IDs are missing form the gis mapping
    missing_ids = gis_mapping_nbal_ids_set - unique_ids

    # Append warnings for each missing ID, these are errors
    for idx in missing_ids:
        warnings.append(f"GIS mapping file NBAL ID {idx}: not in the NBAL shapefile.")


    # --- STEP 6: Return results ---
    return {"warnings": warnings, "errors": errors}


def validate_compartment_shapefile(path: str, gis_mapping_compartment_ids: list,headers_required: list, headers_other: list) -> dict:
    """
        Validate MIU shapefile structure and data.

        Returns:
            dict with "errors" (blocking) and "warnings" (non-blocking).
        """

    warnings, errors = [], []

    compartment_id_header = headers_required[0]
    area_header = headers_required[1]
    slope_header = headers_required[2]
    walk_time_header = headers_required[3]
    drive_time_header = headers_required[4]
    costing_model_header = headers_required[5]
    growth_condition_header = headers_required[6]

    # --- STEP 1: Try opening file ---
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        return {"warnings": warnings, "errors": [f"Could not open shapefile: {e}"]}

    # --- STEP 2: Standardize headers ---
    gdf.columns = [c.lower() for c in gdf.columns]

    # --- STEP 3: Check required columns ---
    missing_columns = [c for c in headers_required if c not in gdf.columns]
    if missing_columns:
        return {"warnings": warnings, "errors": [f"Missing required columns: {missing_columns}"]}

    # --- STEP 4: Error checks ---
    # Area must be numeric
    if area_header in gdf.columns:  # area
        # Find non-numeric values
        non_float_area = gdf[~pd.to_numeric(gdf[area_header], errors="coerce").notna()]
        for idx, row in non_float_area.iterrows():
            errors.append(f"Row {idx}: area is not numeric → {row[area_header]}")

        # Convert to numeric safely
        gdf[area_header] = pd.to_numeric(gdf[area_header], errors="coerce")

        # Find negative values
        negative_area = gdf[gdf[area_header] < 0]
        for idx, row in negative_area.iterrows():
            errors.append(f"Row {idx}: area is negative → {row[area_header]}")

    # --- STEP 5: Error checks ---
    # Slope must be numeric
    if slope_header in gdf.columns:  # slope
        # Find non-numeric values
        non_float_slope = gdf[~pd.to_numeric(gdf[slope_header], errors="coerce").notna()]
        for idx, row in non_float_slope.iterrows():
            errors.append(f"Row {idx}: slope is not numeric → {row[slope_header]}")

        # Convert to numeric safely
        gdf[slope_header] = pd.to_numeric(gdf[slope_header], errors="coerce")

        # Find negative values
        negative_slope = gdf[gdf[slope_header] < 0]
        for idx, row in negative_slope.iterrows():
            errors.append(f"Row {idx}: slope is negative → {row[slope_header]}")

        # Find values >= 90
        too_large_slope = gdf[gdf[slope_header] >= 90]
        for idx, row in too_large_slope.iterrows():
            errors.append(f"Row {idx}: slope >= 90 → {row[slope_header]}")

    # --- STEP 6: Error checks ---
    # Walk Time must be numeric
    if walk_time_header in gdf.columns:  # drive_time
        # Find non-numeric values
        non_float_walk_time = gdf[~pd.to_numeric(gdf[walk_time_header], errors="coerce").notna()]
        for idx, row in non_float_walk_time.iterrows():
            errors.append(f"Row {idx}: walk_time is not numeric → {row[walk_time_header]}")

        # Convert to numeric safely
        gdf[walk_time_header] = pd.to_numeric(gdf[walk_time_header], errors="coerce")

        # Find negative values
        negative_walk_time = gdf[gdf[walk_time_header] < 0]
        for idx, row in negative_walk_time.iterrows():
            errors.append(f"Row {idx}: walk_time is negative → {row[walk_time_header]}")

    # --- STEP 7: Error checks ---
    # Drive Time must be numeric
    if drive_time_header in gdf.columns:  # drive_time
        # Find non-numeric values
        non_float_drive_time = gdf[~pd.to_numeric(gdf[drive_time_header], errors="coerce").notna()]
        for idx, row in non_float_drive_time.iterrows():
            errors.append(f"Row {idx}: drive_time is not numeric → {row[drive_time_header]}")

        # Convert to numeric safely
        gdf[drive_time_header] = pd.to_numeric(gdf[drive_time_header], errors="coerce")

        # Find negative values
        negative_drive_time = gdf[gdf[drive_time_header] < 0]
        for idx, row in negative_drive_time.iterrows():
            errors.append(f"Row {idx}: drive_time is negative → {row[drive_time_header]}")

    # --- STEP 8: Error checks ---
    # costing must be numeric
    if costing_model_header in gdf.columns:  # costing
        # Find non-numeric values
        non_float_costing = gdf[~pd.to_numeric(gdf[costing_model_header], errors="coerce").notna()]
        for idx, row in non_float_costing.iterrows():
            errors.append(f"Row {idx}: costing is not numeric → {row[costing_model_header]}")

        # Convert to numeric safely
        gdf[costing_model_header] = pd.to_numeric(gdf[costing_model_header], errors="coerce")

        # Find negative values
        negative_costing = gdf[gdf[costing_model_header] < 0]
        for idx, row in negative_costing.iterrows():
            errors.append(f"Row {idx}: costing is negative → {row[costing_model_header]}")

    # If any errors exist → return immediately
    if errors:
        return {"warnings": warnings, "errors": errors}

    # --- STEP 5: Warning checks ---
    # 5.1 Missing optional fields
    for field in headers_other:
        if field in gdf.columns:
            missing_rows = gdf[gdf[field].isna()]
            for idx, _ in missing_rows.iterrows():
                warnings.append(f"Row {idx}: missing {field}")

    # 5.2 Duplicate NBAL IDs
    if compartment_id_header in gdf.columns:
        duplicates = gdf[gdf.duplicated(subset=[compartment_id_header], keep=False)]
        for idx, row in duplicates.iterrows():
            warnings.append(f"Row {idx}: duplicate compartment_id -> {row[compartment_id_header]}")


    # 5.3 Find rows where ANY required column is empty/NaN
    for col in headers_required:
        # Find rows where this column is NaN or just whitespace
        missing_rows = gdf[gdf[col].isna() | (gdf[col].astype(str).str.strip() == "")]

        # Collect warnings per row
        for idx, row in missing_rows.iterrows():
            warnings.append(f"Row {idx}: missing value(s) in required columns → {row[headers_required].to_dict()}")

    # ---Find if the compartment id's are in the gis mapping or if any missing
    # Get all unique non-null values from the gdf column
    unique_ids = set(gdf[compartment_id_header].dropna().str.lower().unique())

    # Ensure gis_mapping_compartment_ids is unique too
    gis_mapping_compartment_ids_set = set(gis_mapping_compartment_ids)

    # Find which IDs are missing
    missing_ids = gis_mapping_compartment_ids_set - unique_ids

    # Append warnings for each missing ID
    for idx in missing_ids:
        warnings.append(f"GIS mapping file COMPARTMENT ID {idx}: not in the compartment shapefile.")

    # --- STEP 6: Return results ---
    return {"warnings": warnings, "errors": errors}


def validate_gis_mapping_shapefile(path: str, headers_required: list, headers_other: list) -> dict:
    """
        Validate MIU shapefile structure and data.

        Returns:
            dict with "errors" (blocking) and "warnings" (non-blocking).
        """

    warnings, errors = [], []

    nbal_id_header = headers_required[0]
    miu_id_header = headers_required[1]
    compartment_id_header = headers_required[2]
    area_header = headers_required[3]

    # --- STEP 1: Try opening file ---
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        return {"warnings": warnings, "errors": [f"Could not open shapefile: {e}"]}

    # --- STEP 2: Standardize headers ---
    gdf.columns = [c.lower() for c in gdf.columns]

    # --- STEP 3: Check required columns ---
    missing_columns = [c for c in headers_required if c not in gdf.columns]
    if missing_columns:
        return {"warnings": warnings, "errors": [f"Missing required columns: {missing_columns}"]}

    # --- STEP 4: Error checks ---
    # Area must be numeric
    if area_header in gdf.columns:  # area
        # Find non-numeric values
        non_float_area = gdf[~pd.to_numeric(gdf[area_header], errors="coerce").notna()]
        for idx, row in non_float_area.iterrows():
            errors.append(f"Row {idx}: area is not numeric → {row[area_header]}")

        # Convert to numeric safely
        gdf[area_header] = pd.to_numeric(gdf[area_header], errors="coerce")

        # Find negative values
        negative_area = gdf[gdf[area_header] < 0]
        for idx, row in negative_area.iterrows():
            errors.append(f"Row {idx}: area is negative → {row[area_header]}")

    # --- HIERARCHY RULES CHECK ---
    for idx, row in gdf.iterrows():
        comp = row.get(compartment_id_header)
        miu = row.get(miu_id_header)
        nbal = row.get(nbal_id_header)

        # Rule 1: NBAL requires MIU and COMPARTMENT
        if pd.notna(nbal) and (pd.isna(miu) or pd.isna(comp)):
            errors.append(
                f"Row {idx}: nbal_id exists ({nbal}) but missing miu_id or compartment_id"
            )

        # Rule 2: MIU requires COMPARTMENT
        if pd.notna(miu) and pd.isna(comp):
            errors.append(
                f"Row {idx}: miu_id exists ({miu}) but missing compartment_id"
            )

        # Rule 3: Can't have MIU or NBAL without COMPARTMENT
        if pd.isna(comp) and (pd.notna(miu) or pd.notna(nbal)):
            errors.append(
                f"Row {idx}: compartment_id missing but miu_id/nbal_id present"
            )

        # (Optional redundancy check for clarity)
        if pd.notna(nbal) and pd.notna(miu) and pd.isna(comp):
            errors.append(
                f"Row {idx}: nbal_id and miu_id exist but missing compartment_id"
            )

    # If any errors exist → return immediately
    if errors:
        return {"warnings": warnings, "errors": errors}

    # --- STEP 5: Warning checks ---
    # 5.1 Missing optional fields
    for field in headers_other:
        if field in gdf.columns:
            missing_rows = gdf[gdf[field].isna()]
            for idx, _ in missing_rows.iterrows():
                warnings.append(f"Row {idx}: missing {field}")

    # 5.2 Duplicate ID Checks
    if compartment_id_header in gdf.columns:
        # Case 1: Compartment-only duplicates (no NBAL/MIU)
        subset_cols = [compartment_id_header]
        comp_only = gdf[gdf[nbal_id_header].isna() & gdf[miu_id_header].isna()]
        duplicates = comp_only[comp_only.duplicated(subset=subset_cols, keep=False)]
        for idx, row in duplicates.iterrows():
            warnings.append(f"Row {idx}: duplicate COMPARTMENT only -> {row[compartment_id_header]}")

        # Case 2: Compartment + NBAL duplicates (no MIU)
        if nbal_id_header in gdf.columns:
            subset_cols = [compartment_id_header, nbal_id_header]
            comp_nbal = gdf[gdf[miu_id_header].isna()]
            duplicates = comp_nbal[comp_nbal.duplicated(subset=subset_cols, keep=False)]
            for idx, row in duplicates.iterrows():
                warnings.append(f"Row {idx}: duplicate COMPARTMENT+NBAL -> "
                                f"compartment_id={row[compartment_id_header]}, nbal_id={row[nbal_id_header]}")

        # Case 3: Compartment + NBAL + MIU duplicates
        if nbal_id_header in gdf.columns and miu_id_header in gdf.columns:
            subset_cols = [compartment_id_header, nbal_id_header, miu_id_header]
            duplicates = gdf[gdf.duplicated(subset=subset_cols, keep=False)]
            for idx, row in duplicates.iterrows():
                warnings.append(f"Row {idx}: duplicate COMPARTMENT+NBAL+MIU -> "
                                f"compartment_id={row[compartment_id_header]}, "
                                f"nbal_id={row[nbal_id_header]}, "
                                f"miu_id={row[miu_id_header]}")

    # --- STEP 6: Return results ---
    return {"warnings": warnings, "errors": errors}


def validate_miu_linked_species_excel(path: str, headers_required: list) -> dict:
    """
        Validate MIU shapefile structure and data.

        Returns:
            dict with "errors" (blocking) and "warnings" (non-blocking).
        """

    warnings, errors = [], []

    miu_id_header = headers_required[0]
    species_header = headers_required[1]
    idenscode_header = headers_required[2]
    age_header = headers_required[3]

    # --- STEP 1: Try opening file ---
    try:
        df = pd.read_excel(path)
    except Exception as e:
        return {"warnings": warnings, "errors": [f"Could not open excel file: {e}"]}

    # --- STEP 2: Standardize headers ---
    df.columns = [c.lower() for c in df.columns]

    # --- STEP 3: Check required columns ---
    missing_columns = [c for c in headers_required if c not in df.columns]
    if missing_columns:
        return {"warnings": warnings, "errors": [f"Missing required columns: {missing_columns}"]}

    # --- STEP 4: Error checks ---
    # Density must be numeric
    if idenscode_header in df.columns:  # density
        # Find non-numeric values
        non_float_density = df[~pd.to_numeric(df[idenscode_header], errors="coerce").notna()]
        for idx, row in non_float_density.iterrows():
            errors.append(f"Row {idx}: density is not numeric → {row[idenscode_header]}")

        # Convert to numeric safely
        df[idenscode_header] = pd.to_numeric(df[idenscode_header], errors="coerce")

        # Find negative values
        negative_density = df[df[idenscode_header] < 0]
        for idx, row in negative_density.iterrows():
            errors.append(f"Row {idx}: density is negative → {row[idenscode_header]}")


    # If any errors exist → return immediately
    if errors:
        return {"warnings": warnings, "errors": errors}

    # --- STEP 5: Warning checks ---
    # 5.1 Missing optional fields
    for field in headers_required:
        if field in df.columns:
            missing_rows = df[df[field].isna()]
            for idx, _ in missing_rows.iterrows():
                warnings.append(f"Row {idx}: missing {field}")

    # 5.2 check duplicates
    # Find duplicates (across all four columns)
    duplicates = df[df.duplicated(subset=headers_required, keep=False)]

    # Collect warnings
    for idx, row in duplicates.iterrows():
        row_data = {col: row[col] for col in headers_required}
        warnings.append(f"Row {idx}: is a duplicate -> {row_data}")

    # --- STEP 6: Return results ---
    return {"warnings": warnings, "errors": errors}

def validate_nbal_linked_species_excel(path: str, headers_required: list) -> dict:
    """
        Validate MIU shapefile structure and data.

        Returns:
            dict with "errors" (blocking) and "warnings" (non-blocking).
        """

    warnings, errors = [], []

    nbal_id_header = headers_required[0]
    species_header = headers_required[1]
    idenscode_header = headers_required[2]
    age_header = headers_required[3]

    # --- STEP 1: Try opening file ---
    try:
        df = pd.read_excel(path)
    except Exception as e:
        return {"warnings": warnings, "errors": [f"Could not open excel file: {e}"]}

    # --- STEP 2: Standardize headers ---
    df.columns = [c.lower() for c in df.columns]

    # --- STEP 3: Check required columns ---
    missing_columns = [c for c in headers_required if c not in df.columns]
    if missing_columns:
        return {"warnings": warnings, "errors": [f"Missing required columns: {missing_columns}"]}

    # --- STEP 4: Error checks ---
    # density must be numeric
    if idenscode_header in df.columns:  # density
        # Find non-numeric values
        non_float_density = df[~pd.to_numeric(df[idenscode_header], errors="coerce").notna()]
        for idx, row in non_float_density.iterrows():
            errors.append(f"Row {idx}: density is not numeric → {row[idenscode_header]}")

        # Convert to numeric safely
        df[idenscode_header] = pd.to_numeric(df[idenscode_header], errors="coerce")

        # Find negative values
        negative_density = df[df[idenscode_header] < 0]
        for idx, row in negative_density.iterrows():
            errors.append(f"Row {idx}: density is negative → {row[idenscode_header]}")


    # If any errors exist → return immediately
    if errors:
        return {"warnings": warnings, "errors": errors}

    # --- STEP 5: Warning checks ---
    # 5.1 Missing optional fields
    for field in headers_required:
        if field in df.columns:
            missing_rows = df[df[field].isna()]
            for idx, _ in missing_rows.iterrows():
                warnings.append(f"Row {idx}: missing {field}")

    # 5.2 check duplicates
    # Find duplicates (across all four columns)
    duplicates = df[df.duplicated(subset=headers_required, keep=False)]

    # Collect warnings
    for idx, row in duplicates.iterrows():
        row_data = {col: row[col] for col in headers_required}
        warnings.append(f"Row {idx}: is a duplicate -> {row_data}")

    # --- STEP 6: Return results ---
    return {"warnings": warnings, "errors": errors}


def validate_compartment_priorities_csv(path: str, headers_required: list) -> dict:
    """
        Validate MIU shapefile structure and data.

        Returns:
            dict with "errors" (blocking) and "warnings" (non-blocking).
        """

    warnings, errors = [], []

    compt_id_header = headers_required[0]

    # --- STEP 1: Try opening file ---
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"warnings": warnings, "errors": [f"Could not open csv file: {e}"]}

    # --- STEP 2: Standardize headers ---
    df.columns = [c.lower() for c in df.columns]

    # --- STEP 3: Check required columns ---
    missing_columns = [c for c in headers_required if c not in df.columns]
    if missing_columns:
        return {"warnings": warnings, "errors": [f"Missing required columns: {missing_columns}"]}

    # If any errors exist → return immediately
    if errors:
        return {"warnings": warnings, "errors": errors}

    # --- STEP 5: Warning checks ---
    # Replace empty strings with NaN for consistency
    df = df.replace({"": np.nan})

    # Loop through DataFrame to find missing values
    for idx, row in df.iterrows():
        for col in df.columns:
            if pd.isna(row[col]):
                warnings.append(f"Row {idx}: is missing value in column '{col}'")


    # 5.2 check duplicates
    # Find all duplicate rows (excluding the first occurrence)
    duplicates = df[df.duplicated(keep=False)]

    if not duplicates.empty:
        for idx, row in duplicates.iterrows():
            warnings.append(f"Row {idx}: is a duplicate -> {row.to_dict()}")

    # --- STEP 6: Return results ---
    return {"warnings": warnings, "errors": errors}




