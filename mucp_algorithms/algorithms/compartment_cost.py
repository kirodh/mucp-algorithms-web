"""
Purpose: Cost algorithm of the MUCP tool. Check out the algorithm file for more information.
    This code is optimized to use vectorization for speed.
Author: Kirodh Boodhraj

The main costing algorithm needs:
- person days (calculated using the person day normal)
- initial/follow-up cost per day (need to determine if it is an initial or a follow-up)
- initial/follow-up team size (need to determine if it is an initial or a follow-up)
- vehicle cost per day
- person days normal (step before the person days)
- cost per day (daily costs defined by user, summed up and used here)
- fuel cost per miu species (the distributed fuel cost for each species entry in that specific compartment, i.e. if 3 species
    in compartment C1234, then the cost must be divided by 3)

The costing formula is:
cost =
    person days * initial or follow-up cost per day / initial or follow-up team size +
    vehicle cost per day * person days normal +
    cost per day * person days normal +
    fuel cost per miu species


The output of this algorithm is two parts:
- budgets
- costing results

****The budgets:

These are the propagated budgets using simple interest for each year and budget plan. They take the following structure:
the key is the year, and for each year there is a dictionary with the keys:
- plan_1
- plan_2
- plan_3
- plan_4

and the values for each of these are the calculated budget for that year i.e.

{2025: {'plan_1': 10000000.0, 'plan_2': 7500000.0, 'plan_3': 5000000.0, 'plan_4': 2500000.0},
    2026: {'plan_1': 11000000.0, 'plan_2': 8250000.0, 'plan_3': 5500000.0, 'plan_4': 2750000.0},
    2027: {'plan_1':....}, ...
    }




****The costing results:
They take the structure of a lost, the list contains 5 items, one for each budget in this order:
- 0 index is optimal budget
- 1 index is budget plan 1
- 2 index is budget plan 2
- 3 index is budget plan 3
- 4 index is budget plan 4

In each list item there is a dictionary, the keys are the year of the simulation and the values are a dataframe.

The dataframe contains the full gis mapping entires linked to the results for that year, i.e. the density, priority etc.

For example, to access the optimal budget, year 2025, 1st row of the dataframe or timestep, use:

costing[0][2025].iloc[0]

which results in:

link_back_id                            2
person_days_factor                  1.242
person_days_normal             303.677464
person_days                    618.397745
cost                        153795.154645
density                               3.5
flow                        170598.515202
cleared_fully_previously            False
priority                             8.18
cleared_now                          True
cleared_fully                       False
nbal_id                        h60b400342
miu_id                       m_h60b400342
compt_id                     c_h60b400342
Name: 0, dtype: object

"""

import pandas as pd
import numpy as np
from .compartment_priority import get_priorities
from .compartment_density import calculate_species_density
from .compartment_flow import calculate_flow
from .compartment_person_days import calculate_normal_person_days, calculate_adjusted_person_days


"""
Costing helper functions and algorithm below
"""

# -------------------------------
# Prioritization
# -------------------------------

# find prioritization values per compartment
def get_prioritization(prioritization_model_data, categories):
    return get_priorities(prioritization_model_data, categories)


# attach prioritization values to compartments
def attach_prioritization(full_df: pd.DataFrame, timestep_df: pd.DataFrame) -> pd.Series:
    """
    Attach priority values from full_df to timestep_df based on link_back_id.

    Args:
        full_df: DataFrame containing 'link_back_id' and 'priority'
        timestep_df: DataFrame containing 'link_back_id'

    Returns:
        pd.Series of priority values aligned with timestep_df
    """
    merged = timestep_df.merge(
        full_df[["link_back_id", "prioritization"]],
        on="link_back_id",
        how="left"
    )
    return merged["prioritization"]


# -------------------------------
# Budget
# -------------------------------

# calculate all the budget values for the simulation period, one of the main outputs
def propagate_budgets_for_all_years(start_year: int, years_to_run: int, budget_1: float, budget_2: float, budget_3: float, budget_4: float, escalation_1: float, escalation_2: float, escalation_3: float, escalation_4: float):
    """
    Propagate budgets using simple interest per year. FV = PV * (1 + i*n)

    Parameters:
        start_year (int): First year (treated as year 1, n=0).
        years_to_run (int): How many years to calculate.
        budgets (dict): e.g. {"plan_1": 1000, "plan_2": 2000, ...}
        escalations (dict): e.g. {"plan_1": 20, "plan_2": 10, ...}  # percentages

    Returns:
        dict: {year: {"plan_1": value, "plan_2": value, ...}, ...}
    """
    # Step 1: construct the dicts
    budget_plans = {
        "plan_1": budget_1,
        "plan_2": budget_2,
        "plan_3": budget_3,
        "plan_4": budget_4,
    }

    escalation_plans = {
        "plan_1": escalation_1,
        "plan_2": escalation_2,
        "plan_3": escalation_3,
        "plan_4": escalation_4,
    }

    results = {}
    for year_offset in range(years_to_run):
        year = start_year + year_offset
        results[year] = {}
        for plan, pv in budget_plans.items():
            i = escalation_plans[plan] / 100.0  # convert % -> decimal
            fv = pv * (1 + i * year_offset)  # simple interest
            results[year][plan] = fv
    return results


# -------------------------------
# Merge and split data frames for all data
# -------------------------------

# merge the miu, nbal, compartment and gis mapping files together to make a master df with all data available for all
#   the entries in the gis mapping file
def merge_gis_mapping_compartment_miu_nbal_df(compartment_data, miu_data ,nbal_data, gis_mapping_data):
    # step get all valid entries if they are in the miu, nbal and compartment df's
    # Build sets of valid IDs from the other dataframes
    valid_compt_ids = set(compartment_data["compt_id"].dropna().unique())
    valid_miu_ids = set(miu_data["miu_id"].dropna().unique())
    valid_nbal_ids = set(nbal_data["nbal_id"].dropna().unique())

    # Filter compartments -> must always exist
    gis_mapping_data = gis_mapping_data[gis_mapping_data["compt_id"].isin(valid_compt_ids)]

    # Filter miu -> only check rows where miu_id is not null
    mask_miu = gis_mapping_data["miu_id"].isna() | gis_mapping_data["miu_id"].isin(valid_miu_ids)
    gis_mapping_data = gis_mapping_data[mask_miu]

    # Filter nbal -> only check rows where nbal_id is not null
    mask_nbal = gis_mapping_data["nbal_id"].isna() | gis_mapping_data["nbal_id"].isin(valid_nbal_ids)
    gis_mapping_data = gis_mapping_data[mask_nbal]

    # step 2 merge the initial data into one df,
    # Merge compartments (always exists)
    gis_mapping_data = gis_mapping_data.merge(compartment_data, on="compt_id", how="left", suffixes=("", "_comp"))

    # Merge miu (may be missing)
    gis_mapping_data = gis_mapping_data.merge(miu_data, on="miu_id", how="left", suffixes=("", "_miu"))

    # Merge nbal (may be missing)
    gis_mapping_data = gis_mapping_data.merge(nbal_data, on="nbal_id", how="left", suffixes=("", "_nbal"))

    return gis_mapping_data


# split the master df, into compartment only, compartment + miu only and compartment + miu + nbal rows only
def split_expanded_gis_mapping_df(df):
    # Drop *all* geometry-related columns
    df = df.drop(columns=[c for c in df.columns if "geometry" in c], errors="ignore")

    # --- 1. Only compartment-level rows ---
    comp_only_df = df[
        (df["miu_id"].isna() & df["nbal_id"].isna()) | (df.get("stage") == -1)
    ].copy()

    # --- 2. Compartment + MIU only ---
    comp_miu_df = df[
        ((df["miu_id"].notna()) & df["nbal_id"].isna()) | (df.get("stage") == 0)
    ].copy()

    # --- 3. Everything else (typically includes nbal-level rows) ---
    other_df = df.drop(comp_only_df.index.union(comp_miu_df.index)).copy()

    return comp_only_df, comp_miu_df, other_df


# merge in the tree species data on the master df
def merge_tree_species_data(master_df, miu_linked_species_data, nbal_linked_species_data, species_data, calculate_flow_boolean):
    # Example: comp_miu_df has one row per compartment/miu/nbal combination
    # miu_linked_species_data has multiple species per miu
    # nbal_linked_species_data has multiple species per nbal

    # Step 1: Separate rows that are MIU only vs MIU + NBAL
    miu_only_df = master_df[master_df['nbal_id'].isna()]
    nbal_miu_df = master_df[master_df['nbal_id'].notna()]

    # Step 2: Merge MIU-only rows with miu_linked_species_data
    miu_expanded = miu_only_df.merge(
        miu_linked_species_data,
        on='miu_id',
        how='inner'  # only keep species that exist
    )

    # Step 3: Merge MIU+NBAL rows with nbal_linked_species_data
    nbal_expanded = nbal_miu_df.merge(
        nbal_linked_species_data,
        on='nbal_id',
        how='inner'
    )

    # Step 4: Combine both expanded dataframes
    expanded_df = pd.concat([miu_expanded, nbal_expanded], ignore_index=True)

    # Step 5: Merge in species-specific attributes
    expanded_df = expanded_df.merge(
        species_data,
        left_on="species",
        right_on="species_name",
        how="left"
    )

    # Step 6: If flow calculation requested
    if calculate_flow_boolean:
        # Define conditions for vectorized flow factor selection
        condlist = [
            expanded_df["age"] == "young",
            expanded_df["age"] == "seedling",
            expanded_df["age"] == "coppice",
            expanded_df["age"].isin(["mature", "adult", "mixed"])
        ]

        choicelist = [
            expanded_df["flow_young"],
            expanded_df["flow_seedling"],
            expanded_df["flow_coppice"],
            np.where(
                expanded_df["grow_con"] == "sub-optimal",
                expanded_df["flow_sub_optimal"],
                expanded_df["flow_optimal"]
            )
        ]

        # Vectorized selection
        expanded_df["flow_factor"] = np.select(
            condlist,
            choicelist,
            default=np.where(
                expanded_df["grow_con"] == "sub-optimal",
                expanded_df["flow_sub_optimal"],
                expanded_df["flow_optimal"]
            )
        )

        # Drop the extra flow columns since only flow_factor is needed
        expanded_df = expanded_df.drop(
            columns=["flow_optimal", "flow_sub_optimal", "flow_young", "flow_seedling", "flow_coppice", 'id', 'genus', 'english_name', 'afrikaans_name', 'wc', 'nc', 'kzn', 'gtg', 'mpl', 'fs', 'ec', 'lmp', 'nw'],
            errors="ignore"
        )

    # Step 7: Optional: sort by compartment and species for readability
    expanded_df = expanded_df.sort_values(by=['compt_id', 'species']).reset_index(drop=True)

    # print(expanded_df)
    return expanded_df


#  merge in the cost model data into master df
def merge_cost_model_data(master_df,costing_df):
    costing_df = costing_df.rename(columns={"Costing Model Name": "cost_model"})

    # Keep only the relevant columns from costing_df
    costing_cols = [
        "cost_model",
        "Initial Team Size",
        "Initial Cost/Day",
        "Follow-up Team Size",
        "Follow-up Cost/Day",
        "Vehicle Cost/Day",
        "Fuel Cost/Hour",
        "Maintenance Level",
        "Cost/Day"
    ]

    costing_subset = costing_df[costing_cols]

    # Merge into expanded_df
    return master_df.merge(
        costing_subset,
        on="cost_model",
        how="left"  # keep all expanded_df rows, add costing data
    )


#  merge in the prioritization data in the master df
def merge_prioritization_data(master_df, prioritization_df):
    return master_df.merge(
        prioritization_df,
        on="compt_id",
        how="left"  # keep all expanded_df rows, add costing data
    )


# -------------------------------
# Slope factor
# -------------------------------

# get the slope factor and set in master df
def set_slope_factor(df):
    # Define conditions (based on slope ranges)
    conditions = [
        df['slope'] >= 51,
        df['slope'] >= 41,
        df['slope'] >= 31,
        df['slope'] >= 21,
        df['slope'] >= 11,
        df['slope'] >= 0
    ]

    # Corresponding slope factors
    choices = [2, 1.8, 1.6, 1.4, 1.2, 1]

    # Apply vectorized selection
    # df['slope_factor'] = np.select(conditions, choices, default=1)
    return np.select(conditions, choices, default=1)


# -------------------------------
# Treatment selection
# -------------------------------

# determine treatment and then set it in the master df
def treatment_selection(master_df: pd.DataFrame, norms: pd.DataFrame) -> pd.DataFrame:
    """
    Assign treatment_method to master_df based on norms rules.
    """

    # Extract unique densities once
    unique_densities = norms["density"].dropna().unique()
    unique_densities = np.sort(unique_densities)

    # Helper: find closest density
    def closest_density(value):
        return unique_densities[np.abs(unique_densities - value).argmin()]

    # Priority order for selection
    def choose_method(candidates, age, terrain):
        methods = candidates["treatment_method"].tolist()

        if age == "adult":
            if terrain == "landscape" and "ring bark" in methods:
                return "ring bark"
            if terrain == "riparian" and "felling" in methods:
                return "felling"
            if "bark strip" in methods:
                return "bark strip"

        if "lopping / pruning" in methods:
            return "lopping / pruning"

        return methods[0] if methods else None

    def get_method(row):
        # Pick closest density instead of exact match
        closest = closest_density(row["idenscode"])

        # Initial filter
        mask = (
            (norms["density"] == closest) &
            (norms["growth_form"].str.lower() == row["growth_form"].lower()) &
            (norms["size_class"].str.lower() == row["age"].lower()) &
            (norms["terrain"].str.lower() == row["riparian_c"].lower())
        )
        candidates = norms[mask]

        # If no matches -> drop density filter
        if candidates.empty:
            mask = (
                (norms["growth_form"].str.lower() == row["growth_form"].lower()) &
                (norms["size_class"].str.lower() == row["age"].lower()) &
                (norms["terrain"].str.lower() == row["riparian_c"].lower())
            )
            candidates = norms[mask]

        # If still no matches -> return None
        if candidates.empty:
            return None

        # If only one -> take it
        if len(candidates) == 1:
            return candidates.iloc[0]["treatment_method"]

        # Otherwise apply priority rules
        return choose_method(candidates, row["age"], row["riparian_c"])

    # Apply to each row of master_df
    master_df = master_df.copy()
    master_df["treatment_method"] = master_df.apply(get_method, axis=1)

    return master_df


# -------------------------------
# Cost Support functions
# -------------------------------

# assign unique ids to entries in gis mapping so that the timesteps resulting data can have an easy link back id to the
#   gis mapping file
def assign_unique_ids(*dfs):
    """
    Assign globally unique link_back_id across multiple DataFrames.
    """
    all_dfs = []
    counter = 1

    for df in dfs:
        df = df.copy()
        n_rows = len(df)
        df["link_back_id"] = range(counter, counter + n_rows)
        counter += n_rows
        all_dfs.append(df)

    return all_dfs


# -------------------------------
# Person days Algorithms
# -------------------------------

# calculate the person days
def attach_person_day_factor(full_df: pd.DataFrame, density, norms: pd.DataFrame) -> pd.Series:
    """
    Vectorized: return a Series of ppd values for comp_miu_df
    using closest density + categorical matches.
    """

    # Step 1: map closest density
    unique_densities = np.sort(norms["density"].dropna().unique())
    vals = density.values

    idx = np.searchsorted(unique_densities, vals)
    idx = np.clip(idx, 1, len(unique_densities)-1)

    left = unique_densities[idx - 1]
    right = unique_densities[idx]
    matched_density = np.where(
        np.abs(vals - left) < np.abs(vals - right),
        left,
        right
    )

    # Step 2: merge
    temp = full_df.copy()
    temp["matched_density"] = matched_density

    merged = pd.merge(
        temp,
        norms,
        left_on=["matched_density", "growth_form", "treatment_method", "age", "riparian_c"],
        right_on=["density", "growth_form", "treatment_method", "size_class", "terrain"],
        how="left"
    )

    # Step 3: deduplicate by link_back_id
    merged = merged.sort_values(by=["link_back_id"])
    merged = merged.drop_duplicates(subset=["link_back_id"], keep="first")

    # Step 4: error check
    if merged["ppd"].isna().any():
        missing = merged.loc[
            merged["ppd"].isna(),
            ["link_back_id", "idenscode", "growth_form", "treatment_method", "age", "riparian_c"]
        ]
        raise ValueError(f"Missing PPD match for:\n{missing}")

    # Return just the ppd Series, aligned to original order
    return merged.set_index("link_back_id").loc[full_df["link_back_id"], "ppd"].reset_index(drop=True)


# -------------------------------
# Cost Algorithms
# -------------------------------

# main cost algorithm
def attach_cost(master_df: pd.DataFrame, timestep_df: pd.DataFrame, follow_up: bool = True) -> pd.Series:
    if follow_up:
        # Merge necessary columns from master_df onto timestep_df
        merged = timestep_df.merge(
            master_df[
                [
                    "link_back_id",
                    "Follow-up Cost/Day",
                    "Follow-up Team Size",
                    "Vehicle Cost/Day",
                    "Cost/Day",
                ]
            ],
            on="link_back_id",
            how="left"
        )

        # Apply formula vectorized
        cost_optimal = (
                (merged["person_days"] * merged["Follow-up Cost/Day"] / merged["Follow-up Team Size"]) +
                (merged["Vehicle Cost/Day"] * merged["person_days_normal"]) +
                (merged["Cost/Day"] * merged["person_days_normal"]) + (master_df["fuel_cost_per_miu"])
        )
    else:
        # Merge necessary columns from master_df onto timestep_df
        merged = timestep_df.merge(
            master_df[
                [
                    "link_back_id",
                    "Initial Cost/Day",
                    "Initial Team Size",
                    "Vehicle Cost/Day",
                    "Cost/Day",
                ]
            ],
            on="link_back_id",
            how="left"
        )

        # Apply formula vectorized
        cost_optimal = (
            (merged["person_days"] * merged["Initial Cost/Day"] / merged["Initial Team Size"]) +
            (merged["Vehicle Cost/Day"] * merged["person_days_normal"]) +
            (merged["Cost/Day"] * merged["person_days_normal"]) + (master_df["fuel_cost_per_miu"])
        )

    return cost_optimal


# clearing rules for compartment
# determine and add cleared now to the timestep
def attach_cleared_now(
    master_df: pd.DataFrame,
    timestep_df: pd.DataFrame,
    year_budget: float,
) -> pd.DataFrame:

    # Merge metadata into timestep_df
    merged = timestep_df.merge(
        master_df[
            [
                "link_back_id", "prioritization" #, "Maintenance Level"
            ]
        ],
        on="link_back_id",
        how="left"
    )

    # Preserve original order
    merged["_orig_order"] = np.arange(len(merged))

    # Start masks
    cleared_now_mask = pd.Series(False, index=merged.index)
    # cleared_fully_mask = merged["cleared_fully_previously"].copy()

    # Only consider not-fully-cleared
    candidates = merged.loc[~merged["cleared_fully_previously"]].copy()

    # Sort by rules: priority ↓, density ↑, cost ↑
    candidates = candidates.sort_values(
        by=["prioritization", "density", "cost"],
        ascending=[False, True, True]
    )

    # Vectorized cumulative allocation
    cumsum_cost = candidates["cost"].cumsum()
    affordable_mask = cumsum_cost <= year_budget

    # Mark all rows up to the first overshoot
    cleared_now_mask.loc[candidates.index] = affordable_mask
    budget_left = year_budget - candidates.loc[affordable_mask, "cost"].sum()

    # Handle the tail (after overshoot) with greedy skipping
    remaining = candidates.loc[~affordable_mask]
    for idx, row in remaining.iterrows():
        cost = row["cost"]
        if cost <= budget_left:
            budget_left -= cost
            cleared_now_mask.at[idx] = True
        if budget_left <= 0:
            break

    # Assign results back
    merged["cleared_now"] = cleared_now_mask

    # Restore original row order
    merged = merged.sort_values("_orig_order").drop(columns="_orig_order")

    # Return Series aligned to timestep_df
    return merged["cleared_now"].reset_index(drop=True)

# determine and add cleared fully to the timestep
def attach_cleared_fully(master_df: pd.DataFrame, timestep_df: pd.DataFrame) -> pd.DataFrame:
    """
    Update cleared_fully in timestep_df:
    - Keep previous cleared_fully_previously values
    - Add new True values where density < Maintenance Level
    """

    # Merge maintenance values onto timestep_df
    merged = timestep_df.merge(
        master_df[["link_back_id", "Maintenance Level"]],
        on="link_back_id",
        how="left"
    )

    # Start with previous values
    cleared_fully = merged["cleared_fully_previously"].copy()

    # Add condition: density < Maintenance Level
    cleared_fully |= merged["density"] < merged["Maintenance Level"]

    # Return updated column
    return cleared_fully


# -------------------------------
# Density Algorithms
# -------------------------------
## directly inserted in costing loops below, equation can be broadcasted into the arrays efficiently


# -------------------------------
# Flow Algorithms
# -------------------------------

# attach the matched Mean Annual Runoff (MAR) to the compartment id in the master df
def attach_runoff_to_compartment(runoff_data: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
    """
        Attach runoff values to the main_df based on matching compt_id.

        Parameters
        ----------
        runoff_data : pd.DataFrame
            DataFrame containing columns ['compt_id', 'runoff'] (and possibly others).
        main_df : pd.DataFrame
            Main DataFrame containing 'compt_id'.

        Returns
        -------
        pd.DataFrame
            main_df with an extra column 'runoff' merged in.
        """
    # Merge on compt_id
    merged_df = main_df.merge(
        runoff_data[["compt_id", "runoff"]],
        on="compt_id",
        how="left"  # keeps all rows in main_df
    )

    return merged_df


# Attach the initial flow values to the initial timestep
def attach_flow_initial_timestep(master_df, current_timestep, follow_up=True):
    # Merge only once on link_back_id
    if follow_up:
        merged = current_timestep.merge(
            master_df[["link_back_id", "runoff", "follow_up_reduction", "flow_factor", "riparian_c", "area"]],
            on="link_back_id",
            how="left"
        )

        # Vectorized call to calculate_flow
        flow_optimal = np.vectorize(calculate_flow)(
            merged["runoff"],  # MAR
            merged["area"],  # area
            merged["density_optimal"],  # density from timestep_df
            merged["follow_up_reduction"],  # density_factor
            merged["flow_factor"],  # flow_reduction_factor
            merged["riparian_c"],  # boolean riparian
        )

    else:
        merged = current_timestep.merge(
            master_df[["link_back_id", "runoff", "initial_reduction", "flow_factor", "riparian_c", "area"]],
            on="link_back_id",
            how="left"
        )

        # Vectorized call to calculate_flow
        flow_optimal = np.vectorize(calculate_flow)(
            merged["runoff"],                  # MAR
            merged["area"],                    # area
            merged["density_optimal"],                 # density from timestep_df
            merged["initial_reduction"],       # density_factor
            merged["flow_factor"],             # flow_reduction_factor
            merged["riparian_c"],                      # boolean riparian
        )


    return flow_optimal


# Attach the flow values to the current timestep, for follow up
def attach_flow(master_df, current_timestep, follow_up=True):
    # Merge only once on link_back_id
    if follow_up:
        merged = current_timestep.merge(
            master_df[["link_back_id", "runoff", "follow_up_reduction", "flow_factor", "riparian_c", "area"]],
            on="link_back_id",
            how="left"
        )

        # Vectorized call to calculate_flow
        flow_optimal = np.vectorize(calculate_flow)(
            merged["runoff"],  # MAR
            merged["area"],  # area
            merged["density"],  # density from timestep_df
            merged["follow_up_reduction"],  # density_factor
            merged["flow_factor"],  # flow_reduction_factor
            merged["riparian_c"],  # boolean riparian
        )

    else:
        merged = current_timestep.merge(
            master_df[["link_back_id", "runoff", "initial_reduction", "flow_factor", "riparian_c", "area"]],
            on="link_back_id",
            how="left"
        )

        # Vectorized call to calculate_flow
        flow_optimal = np.vectorize(calculate_flow)(
            merged["runoff"],                  # MAR
            merged["area"],                    # area
            merged["density"],                 # density from timestep_df
            merged["initial_reduction"],       # density_factor
            merged["flow_factor"],             # flow_reduction_factor
            merged["riparian_c"],                      # boolean riparian
        )


    return flow_optimal


# post process the final results
def postprocess_yearly_results(dicts_full, comp_miu_df, follow_up_df, com_only_df):
    # Extract only needed columns from comp_miu_df & follow_up_df
    id_lookup = pd.concat([
        comp_miu_df[["link_back_id", "nbal_id", "miu_id", "compt_id"]],
        follow_up_df[["link_back_id", "nbal_id", "miu_id", "compt_id"]]
    ]).drop_duplicates("link_back_id")

    for yearly_results in dicts_full:  # loop over yearly_results_o, yearly_results_1, etc.
        for year, df in yearly_results.items():
            # --- Step 1: Align IDs ---
            df = df.merge(id_lookup, on="link_back_id", how="left")

            # --- Step 2: Append comp_only_df rows ---
            new_rows = com_only_df[["nbal_id", "miu_id", "compt_id", "link_back_id"]].copy()
            new_rows["cleared_fully"] = False
            new_rows["cleared_fully_previously"] = False
            new_rows["cleared_now"] = False

            # Fill other numeric columns with 0
            for col in df.columns:
                if col not in new_rows.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        new_rows[col] = 0
                    else:
                        new_rows[col] = np.nan

            # Concat old df + new rows
            df = pd.concat([df, new_rows], ignore_index=True)

            # --- Step 3: Ensure flow column exists ---
            if "flow" not in df.columns:
                df["flow"] = np.nan

            # Save back
            yearly_results[year] = df

    return dicts_full


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

"""
The main costing function loop is below, you need to call this function to get a costing for the MUCP tool.

The example file shows how to read your support and file data, and how to pass those information into the tool.

"""

#### MAIN COSTING FUNCTION
def calculate_budgets(gis_mapping_data, miu_data, nbal_data, compartment_data, miu_linked_species_data, nbal_linked_species_data, compartment_priorities_data, growth_forms, treatment_methods, clearing_norms_df, species, costing_data, budget_plan_1, budget_plan_2, budget_plan_3, budget_plan_4, escalation_plan_1, escalation_plan_2, escalation_plan_3, escalation_plan_4, standard_working_day, working_year_days, start_year, years_to_run, currency, save_results, cost_plan_mappings, categories, prioritization_model_data):

    # step 1 fixed variables
    ## prioritization
    prioritization_df = get_prioritization(prioritization_model_data, categories)

    # will flow be calculated?
    calculate_flow_boolean = "runoff" in compartment_priorities_data.columns

    # split the clearing norms into initial and followup
    clearing_norms_df_followup = clearing_norms_df[clearing_norms_df["process"] == "follow-up"].copy()
    clearing_norms_df_initial = clearing_norms_df[clearing_norms_df["process"] == "initial"].copy()

    # step 2 merge the gis mapping with mui, nbal and compt
    master_data = merge_gis_mapping_compartment_miu_nbal_df(compartment_data, miu_data, nbal_data, gis_mapping_data)

    # step 3 split master data into sections appropriate for running separately based on mui, nbal and compt
    comp_only_df, comp_miu_df, follow_up_df = split_expanded_gis_mapping_df(master_data)

    # step 4 cost mapping, map cost code to the actual name of the model:
    comp_miu_df['cost_model'] = comp_miu_df['costing'].map(cost_plan_mappings)
    follow_up_df['cost_model'] = follow_up_df['costing'].map(cost_plan_mappings)

    # step 5 reparian mapping:
    mapping = {"l": "landscape", "r": "riparian"}
    comp_miu_df["riparian_c"] = comp_miu_df["riparian_c"].map(mapping)
    follow_up_df["riparian_c"] = follow_up_df["riparian_c"].map(mapping)

    # step 6 set the slope factor
    comp_miu_df["slope_factor"] = set_slope_factor(comp_miu_df)
    follow_up_df["slope_factor"] = set_slope_factor(follow_up_df)

    # step 7 insert tree species data
    comp_miu_df = merge_tree_species_data(comp_miu_df,miu_linked_species_data, nbal_linked_species_data,species,calculate_flow_boolean)
    follow_up_df = merge_tree_species_data(follow_up_df,miu_linked_species_data, nbal_linked_species_data,species,calculate_flow_boolean)

    # step 8 insert the cost model parameters
    comp_miu_df = merge_cost_model_data(comp_miu_df,costing_data)
    follow_up_df = merge_cost_model_data(follow_up_df,costing_data)

    # step 9 merge the priority into the data
    comp_miu_df = merge_prioritization_data(comp_miu_df, prioritization_df)
    follow_up_df = merge_prioritization_data(follow_up_df, prioritization_df)

    # step 10 determine treatment method
    comp_miu_df = treatment_selection(comp_miu_df, clearing_norms_df_initial)
    follow_up_df = treatment_selection(follow_up_df, clearing_norms_df_followup)

    # step 11 add on the fuel cost per miu
    # Count how many miu_id per compt_id
    miu_counts_initial = comp_miu_df.groupby("compt_id")["miu_id"].transform("nunique")
    miu_counts_followup = follow_up_df.groupby("compt_id")["miu_id"].transform("nunique")
    # Apply cost formula, its a fixed cost
    ## (Fuel cost per Hour * Drive Time) / number of miu's per compartment / 10
    comp_miu_df["fuel_cost_per_miu"] = (comp_miu_df["Fuel Cost/Hour"] * comp_miu_df["drive_time"] / (miu_counts_initial * 10))
    follow_up_df["fuel_cost_per_miu"] = (follow_up_df["Fuel Cost/Hour"] * follow_up_df["drive_time"] / (miu_counts_followup * 10))

    # step 12 drop unneccesary columns
    unnecessary_columns_to_drop = ["stage", "area_miu", "costing", "area_ha", "slope"]
    comp_miu_df = comp_miu_df.drop(columns=unnecessary_columns_to_drop, errors="ignore")
    follow_up_df = follow_up_df.drop(columns=unnecessary_columns_to_drop, errors="ignore")

    # step 13 propagate budgets
    budgets = propagate_budgets_for_all_years(start_year, years_to_run, budget_plan_1, budget_plan_2, budget_plan_3, budget_plan_4, escalation_plan_1, escalation_plan_2, escalation_plan_3, escalation_plan_4)

    # step 14 insert flow mean annual runoff if its to be calculated
    comp_miu_df = attach_runoff_to_compartment(compartment_priorities_data, comp_miu_df)
    follow_up_df = attach_runoff_to_compartment(compartment_priorities_data, follow_up_df)

    # step 15 assign unique ids to link back to the dfs
    comp_only_df, comp_miu_df, follow_up_df = assign_unique_ids(
        comp_only_df, comp_miu_df, follow_up_df
    )

    # print(comp_only_df)
    # print(comp_only_df.columns)
    # print(comp_only_df.iloc[0])
    # print(comp_miu_df)
    # print(comp_miu_df.columns)
    # print(comp_miu_df.iloc[0])
    # print(follow_up_df)
    # print(follow_up_df.columns)
    # print(follow_up_df.iloc[0])



    #########################################################################################################
    # main costing algorithm
    #########################################################################################################
    # deal with year 1

    # budget dictionaries
    yearly_results_o = {} # optimal
    yearly_results_1 = {}
    yearly_results_2 = {}
    yearly_results_3 = {}
    yearly_results_4 = {}

    # --- Timestep 0 (initial year) ---
    initial_year = start_year

    # ******************
    ### INITIAL
    # ******************
    timestep_df_0_miu_initial = pd.DataFrame({
        "link_back_id": comp_miu_df["link_back_id"],
    })

    # 1. person days
    timestep_df_0_miu_initial["person_days_factor"] = attach_person_day_factor(comp_miu_df,comp_miu_df["idenscode"], clearing_norms_df_initial)
    # normal person days
    timestep_df_0_miu_initial["person_days_normal"] = calculate_normal_person_days(timestep_df_0_miu_initial["person_days_factor"], comp_miu_df["area"])
    # adjusted person days:
    timestep_df_0_miu_initial["person_days"] = calculate_adjusted_person_days(timestep_df_0_miu_initial["person_days_normal"], comp_miu_df["walk_time"], comp_miu_df["drive_time"], comp_miu_df["slope_factor"], standard_working_day)

    # 2. costs
    # cost_optimal
    timestep_df_0_miu_initial["cost_optimal"] = attach_cost(comp_miu_df,timestep_df_0_miu_initial,follow_up=False)

    # 3. densities
    # density_optimal is for assuming all densities were done
    # density is the compartments that werent cleared are densified, and the cleared ones are reduced
    timestep_df_0_miu_initial["density_optimal"] = calculate_species_density(comp_miu_df["idenscode"], -1*comp_miu_df["initial_reduction"])

    # 4. flow
    if calculate_flow_boolean:
        timestep_df_0_miu_initial["flow_optimal"] = attach_flow_initial_timestep(comp_miu_df, timestep_df_0_miu_initial, follow_up=False)


    # ******************
    ### FOLLOW UP
    # ******************

    # construct the initial time step follow-up calculations
    timestep_df_0_nbal_followup = pd.DataFrame({
        "link_back_id": follow_up_df["link_back_id"],
    })

    # 1. person days
    timestep_df_0_nbal_followup["person_days_factor"] = attach_person_day_factor(follow_up_df, follow_up_df["idenscode"], clearing_norms_df_followup)
    # normal person days
    timestep_df_0_nbal_followup["person_days_normal"] = calculate_normal_person_days(timestep_df_0_nbal_followup["person_days_factor"], follow_up_df["area"])
    # adjusted person days:
    timestep_df_0_nbal_followup["person_days"] = calculate_adjusted_person_days(timestep_df_0_nbal_followup["person_days_normal"], follow_up_df["walk_time"], follow_up_df["drive_time"], follow_up_df["slope_factor"], standard_working_day)

    # 2. costs
    # cost_optimal
    timestep_df_0_nbal_followup["cost_optimal"] = attach_cost(follow_up_df,timestep_df_0_nbal_followup)

    # 3. densities
    # density_optimal is for assuming all densities were done
    # density is the compartments that werent cleared are densified, ad the cleared ones are reduced
    timestep_df_0_nbal_followup["density_optimal"] = calculate_species_density(follow_up_df["idenscode"], -1*follow_up_df["initial_reduction"])

    # 4. flow
    if calculate_flow_boolean:
        timestep_df_0_nbal_followup["flow_optimal"] = attach_flow_initial_timestep(follow_up_df, timestep_df_0_nbal_followup)


    # ******************
    ### NEXT PHASE, COMBINATION of the initial and follow up for first time step
    # ******************
    # 1. Combine timestep DataFrames, for each budget
    timestep_df_0_optimal = pd.concat(
        [timestep_df_0_miu_initial, timestep_df_0_nbal_followup],
        ignore_index=True  # resets index, but keeps link_back_id intact
    )

    # 2. Remove "_optimal" from column names
    timestep_df_0_optimal = timestep_df_0_optimal.rename(
        columns=lambda c: c.replace("_optimal", "")
    ).copy()

    # 4. Create copies for timestep_df_0_1 through timestep_df_0_4
    timestep_df_0_1 = timestep_df_0_optimal.copy()
    timestep_df_0_2 = timestep_df_0_optimal.copy()
    timestep_df_0_3 = timestep_df_0_optimal.copy()
    timestep_df_0_4 = timestep_df_0_optimal.copy()


    # 5. combine the initial master df into one, as all will be follow up from now on after calculating the first time step
    master_data = pd.concat(
        # [comp_only_df, comp_miu_df, follow_up_df], # remember to add back on the compartment only entries
        [comp_miu_df, follow_up_df],
        ignore_index=True  # resets index, but keeps link_back_id intact
    )


    # 6. Add a cleared fully column, initial cleared fully previously variable
    timestep_df_0_optimal["cleared_fully_previously"] = False
    timestep_df_0_1["cleared_fully_previously"] = False
    timestep_df_0_2["cleared_fully_previously"] = False
    timestep_df_0_3["cleared_fully_previously"] = False
    timestep_df_0_4["cleared_fully_previously"] = False


    # 7. add the prioritizations on:
    timestep_df_0_optimal["priority"] = attach_prioritization(master_data, timestep_df_0_1) # can use same as any plan
    timestep_df_0_1["priority"] = attach_prioritization(master_data, timestep_df_0_1)
    timestep_df_0_2["priority"] = attach_prioritization(master_data, timestep_df_0_2)
    timestep_df_0_3["priority"] = attach_prioritization(master_data, timestep_df_0_3)
    timestep_df_0_4["priority"] = attach_prioritization(master_data, timestep_df_0_4)


    # 8. do the clearing and the final values for person days, density etc for each budget
    # clearing now booleans
    timestep_df_0_optimal["cleared_now"] = True # for optimal all is cleared
    timestep_df_0_1["cleared_now"] = attach_cleared_now(master_data, timestep_df_0_1, budgets[np.int64(initial_year)]['plan_1'])
    timestep_df_0_2["cleared_now"] = attach_cleared_now(master_data, timestep_df_0_2, budgets[np.int64(initial_year)]['plan_2'])
    timestep_df_0_3["cleared_now"] = attach_cleared_now(master_data, timestep_df_0_3, budgets[np.int64(initial_year)]['plan_3'])
    timestep_df_0_4["cleared_now"] = attach_cleared_now(master_data, timestep_df_0_4, budgets[np.int64(initial_year)]['plan_4'])


    # 9. density per budget
    # Pick factor depending on cleared_now
    density_factors_initial_1 = np.where(timestep_df_0_1["cleared_now"], -1*master_data["initial_reduction"], master_data["densification"])
    density_factors_initial_2 = np.where(timestep_df_0_2["cleared_now"], -1*master_data["initial_reduction"], master_data["densification"])
    density_factors_initial_3 = np.where(timestep_df_0_3["cleared_now"], -1*master_data["initial_reduction"], master_data["densification"])
    density_factors_initial_4 = np.where(timestep_df_0_4["cleared_now"], -1*master_data["initial_reduction"], master_data["densification"])
    # Compute density
    timestep_df_0_1["density"] = calculate_species_density(master_data["idenscode"], density_factors_initial_1)
    timestep_df_0_2["density"] = calculate_species_density(master_data["idenscode"], density_factors_initial_2)
    timestep_df_0_3["density"] = calculate_species_density(master_data["idenscode"], density_factors_initial_3)
    timestep_df_0_4["density"] = calculate_species_density(master_data["idenscode"], density_factors_initial_4)


    # 10. person days per budget
    # adjusted person days:
    timestep_df_0_1["person_days"] = np.where(timestep_df_0_1["cleared_now"], timestep_df_0_1["person_days"], 0)
    timestep_df_0_2["person_days"] = np.where(timestep_df_0_2["cleared_now"], timestep_df_0_2["person_days"], 0)
    timestep_df_0_3["person_days"] = np.where(timestep_df_0_3["cleared_now"], timestep_df_0_3["person_days"], 0)
    timestep_df_0_4["person_days"] = np.where(timestep_df_0_4["cleared_now"], timestep_df_0_4["person_days"], 0)


    # 11. cost per budget
    timestep_df_0_1["cost"] = np.where(timestep_df_0_1["cleared_now"], timestep_df_0_1["cost"], 0)
    timestep_df_0_2["cost"] = np.where(timestep_df_0_2["cleared_now"], timestep_df_0_2["cost"], 0)
    timestep_df_0_3["cost"] = np.where(timestep_df_0_3["cleared_now"], timestep_df_0_3["cost"], 0)
    timestep_df_0_4["cost"] = np.where(timestep_df_0_4["cleared_now"], timestep_df_0_4["cost"], 0)


    # 12. flow per budget
    if calculate_flow_boolean:
        timestep_df_0_1["flow"] = np.where(timestep_df_0_1["cleared_now"], timestep_df_0_1["flow"], 0)
        timestep_df_0_2["flow"] = np.where(timestep_df_0_2["cleared_now"], timestep_df_0_2["flow"], 0)
        timestep_df_0_3["flow"] = np.where(timestep_df_0_3["cleared_now"], timestep_df_0_3["flow"], 0)
        timestep_df_0_4["flow"] = np.where(timestep_df_0_4["cleared_now"], timestep_df_0_4["flow"], 0)


    # 13. cleared fully booleans
    timestep_df_0_optimal["cleared_fully"] = attach_cleared_fully(master_data,timestep_df_0_optimal)
    timestep_df_0_1["cleared_fully"] = attach_cleared_fully(master_data,timestep_df_0_1)
    timestep_df_0_2["cleared_fully"] = attach_cleared_fully(master_data,timestep_df_0_2)
    timestep_df_0_3["cleared_fully"] = attach_cleared_fully(master_data,timestep_df_0_3)
    timestep_df_0_4["cleared_fully"] = attach_cleared_fully(master_data,timestep_df_0_4)


    # 14. Save as first timestep df , cut out the right columns and rename
    yearly_results_o[initial_year] = timestep_df_0_optimal
    yearly_results_1[initial_year] = timestep_df_0_1
    yearly_results_2[initial_year] = timestep_df_0_2
    yearly_results_3[initial_year] = timestep_df_0_3
    yearly_results_4[initial_year] = timestep_df_0_4


    # timestep one complete, loop through year 2 etc.
    # ----------------------------------------
    # YEAR 2 until END
    # ----------------------------------------
    # step 0: --- Timestep 1+ (follow-up years) ---
    prev_year_df_optimal = timestep_df_0_optimal.copy()
    prev_year_df_1 = timestep_df_0_1.copy()
    prev_year_df_2 = timestep_df_0_2.copy()
    prev_year_df_3 = timestep_df_0_3.copy()
    prev_year_df_4 = timestep_df_0_4.copy()

    for year in range(start_year + 1, start_year + years_to_run):
        # step 1: create stub of new timestep
        # Create timestep_now with selected and renamed columns
        timestep_now_optimal = timestep_df_0_optimal.loc[:, ["link_back_id", "priority", "cleared_fully"]].rename(columns={"cleared_fully": "cleared_fully_previously"}).copy()
        timestep_now_1 = timestep_df_0_1.loc[:, ["link_back_id", "priority", "cleared_fully"]].rename(columns={"cleared_fully": "cleared_fully_previously"}).copy()
        timestep_now_2 = timestep_df_0_2.loc[:, ["link_back_id", "priority", "cleared_fully"]].rename(columns={"cleared_fully": "cleared_fully_previously"}).copy()
        timestep_now_3 = timestep_df_0_3.loc[:, ["link_back_id", "priority", "cleared_fully"]].rename(columns={"cleared_fully": "cleared_fully_previously"}).copy()
        timestep_now_4 = timestep_df_0_4.loc[:, ["link_back_id", "priority", "cleared_fully"]].rename(columns={"cleared_fully": "cleared_fully_previously"}).copy()


        # step 2: align the previous timestep data to the master_data, this is to ensure when you slice data they correspond to the same rows
        prev_year_df_optimal = (prev_year_df_optimal.set_index("link_back_id").reindex(master_data["link_back_id"]).reset_index())
        prev_year_df_1 = (prev_year_df_1.set_index("link_back_id").reindex(master_data["link_back_id"]).reset_index())
        prev_year_df_2 = (prev_year_df_2.set_index("link_back_id").reindex(master_data["link_back_id"]).reset_index())
        prev_year_df_3 = (prev_year_df_3.set_index("link_back_id").reindex(master_data["link_back_id"]).reset_index())
        prev_year_df_4 = (prev_year_df_4.set_index("link_back_id").reindex(master_data["link_back_id"]).reset_index())


        # step 3: ppd
        # calculate the person days factor
        timestep_now_optimal["person_days_factor"] = attach_person_day_factor(master_data, prev_year_df_optimal["density"], clearing_norms_df_followup)
        timestep_now_1["person_days_factor"] = attach_person_day_factor(master_data, prev_year_df_1["density"], clearing_norms_df_followup)
        timestep_now_2["person_days_factor"] = attach_person_day_factor(master_data, prev_year_df_2["density"], clearing_norms_df_followup)
        timestep_now_3["person_days_factor"] = attach_person_day_factor(master_data, prev_year_df_3["density"], clearing_norms_df_followup)
        timestep_now_4["person_days_factor"] = attach_person_day_factor(master_data, prev_year_df_4["density"], clearing_norms_df_followup)


        # step 4: normal person days
        timestep_now_optimal["person_days_normal"] = calculate_normal_person_days(timestep_now_optimal["person_days_factor"], master_data["area"])
        timestep_now_1["person_days_normal"] = calculate_normal_person_days(timestep_now_1["person_days_factor"], master_data["area"])
        timestep_now_2["person_days_normal"] = calculate_normal_person_days(timestep_now_2["person_days_factor"], master_data["area"])
        timestep_now_3["person_days_normal"] = calculate_normal_person_days(timestep_now_3["person_days_factor"], master_data["area"])
        timestep_now_4["person_days_normal"] = calculate_normal_person_days(timestep_now_4["person_days_factor"], master_data["area"])


        # step 5: adjusted person days:
        timestep_now_optimal["person_days"] = calculate_adjusted_person_days(timestep_now_optimal["person_days_normal"], master_data["walk_time"], master_data["drive_time"],master_data["slope_factor"], standard_working_day)
        timestep_now_1["person_days"] = calculate_adjusted_person_days(timestep_now_1["person_days_normal"], master_data["walk_time"], master_data["drive_time"],master_data["slope_factor"], standard_working_day)
        timestep_now_2["person_days"] = calculate_adjusted_person_days(timestep_now_2["person_days_normal"], master_data["walk_time"], master_data["drive_time"],master_data["slope_factor"], standard_working_day)
        timestep_now_3["person_days"] = calculate_adjusted_person_days(timestep_now_3["person_days_normal"], master_data["walk_time"], master_data["drive_time"],master_data["slope_factor"], standard_working_day)
        timestep_now_4["person_days"] = calculate_adjusted_person_days(timestep_now_4["person_days_normal"], master_data["walk_time"], master_data["drive_time"],master_data["slope_factor"], standard_working_day)


        # step 6: costs
        # cost_optimal
        timestep_now_optimal["cost"] = attach_cost(master_data, timestep_now_optimal)
        timestep_now_1["cost"] = attach_cost(master_data, timestep_now_1)
        timestep_now_2["cost"] = attach_cost(master_data, timestep_now_2)
        timestep_now_3["cost"] = attach_cost(master_data, timestep_now_3)
        timestep_now_4["cost"] = attach_cost(master_data, timestep_now_4)


        # step 7: assign old densities to the timestep, so it can be used in the costing and then assign new densities over it
        timestep_now_optimal["density"] = timestep_now_optimal["link_back_id"].map(prev_year_df_optimal.set_index("link_back_id")["density"])
        timestep_now_1["density"] = timestep_now_1["link_back_id"].map(prev_year_df_1.set_index("link_back_id")["density"])
        timestep_now_2["density"] = timestep_now_2["link_back_id"].map(prev_year_df_2.set_index("link_back_id")["density"])
        timestep_now_3["density"] = timestep_now_3["link_back_id"].map(prev_year_df_3.set_index("link_back_id")["density"])
        timestep_now_4["density"] = timestep_now_4["link_back_id"].map(prev_year_df_4.set_index("link_back_id")["density"])


        # step 8: do the clearing and the final values for person days, density etc for each budget
        # clearing now booleans
        timestep_now_optimal["cleared_now"] = True  # for optimal all is cleared
        timestep_now_1["cleared_now"] = attach_cleared_now(master_data, timestep_now_1,budgets[np.int64(year)]['plan_1'])
        timestep_now_2["cleared_now"] = attach_cleared_now(master_data, timestep_now_2,budgets[np.int64(year)]['plan_2'])
        timestep_now_3["cleared_now"] = attach_cleared_now(master_data, timestep_now_3,budgets[np.int64(year)]['plan_3'])
        timestep_now_4["cleared_now"] = attach_cleared_now(master_data, timestep_now_4,budgets[np.int64(year)]['plan_4'])


        # step 9: density per budget (assign over the old densities)
        # Pick factor depending on cleared_now
        density_factors_now_optimal = np.where(timestep_now_optimal["cleared_now"], -1*master_data["initial_reduction"],master_data["densification"])
        density_factors_now_1 = np.where(timestep_now_1["cleared_now"], -1*master_data["initial_reduction"],master_data["densification"])
        density_factors_now_2 = np.where(timestep_now_2["cleared_now"], -1*master_data["initial_reduction"],master_data["densification"])
        density_factors_now_3 = np.where(timestep_now_3["cleared_now"], -1*master_data["initial_reduction"],master_data["densification"])
        density_factors_now_4 = np.where(timestep_now_4["cleared_now"], -1*master_data["initial_reduction"],master_data["densification"])
        # Compute density
        timestep_now_optimal["density"] = calculate_species_density(prev_year_df_optimal["density"], density_factors_now_optimal)
        timestep_now_1["density"] = calculate_species_density(prev_year_df_1["density"], density_factors_now_1)
        timestep_now_2["density"] = calculate_species_density(prev_year_df_2["density"], density_factors_now_2)
        timestep_now_3["density"] = calculate_species_density(prev_year_df_3["density"], density_factors_now_3)
        timestep_now_4["density"] = calculate_species_density(prev_year_df_4["density"], density_factors_now_4)


        # step 10: calculate flow:
        if calculate_flow_boolean:
            timestep_now_optimal["flow"] = attach_flow(master_data, timestep_now_optimal)
            timestep_now_1["flow"] = attach_flow(master_data, timestep_now_1)
            timestep_now_2["flow"] = attach_flow(master_data, timestep_now_2)
            timestep_now_3["flow"] = attach_flow(master_data, timestep_now_3)
            timestep_now_4["flow"] = attach_flow(master_data, timestep_now_4)


        # step 11: person days per budget
        # person days adjustment after clearing
        # adjusted person days:
        timestep_now_optimal["person_days"] = np.where(timestep_now_optimal["cleared_now"], timestep_now_optimal["person_days"], 0)
        timestep_now_1["person_days"] = np.where(timestep_now_1["cleared_now"], timestep_now_1["person_days"], 0)
        timestep_now_2["person_days"] = np.where(timestep_now_2["cleared_now"], timestep_now_2["person_days"], 0)
        timestep_now_3["person_days"] = np.where(timestep_now_3["cleared_now"], timestep_now_3["person_days"], 0)
        timestep_now_4["person_days"] = np.where(timestep_now_4["cleared_now"], timestep_now_4["person_days"], 0)


        # step 12: cost per budget
        timestep_now_optimal["cost"] = np.where(timestep_now_optimal["cleared_now"], timestep_now_optimal["cost"], 0)
        timestep_now_1["cost"] = np.where(timestep_now_1["cleared_now"], timestep_now_1["cost"], 0)
        timestep_now_2["cost"] = np.where(timestep_now_2["cleared_now"], timestep_now_2["cost"], 0)
        timestep_now_3["cost"] = np.where(timestep_now_3["cleared_now"], timestep_now_3["cost"], 0)
        timestep_now_4["cost"] = np.where(timestep_now_4["cleared_now"], timestep_now_4["cost"], 0)


        # step 13: flow per budget
        if calculate_flow_boolean:
            timestep_now_optimal["flow"] = np.where(timestep_now_optimal["cleared_now"], timestep_now_optimal["flow"], 0)
            timestep_now_1["flow"] = np.where(timestep_now_1["cleared_now"], timestep_now_1["flow"], 0)
            timestep_now_2["flow"] = np.where(timestep_now_2["cleared_now"], timestep_now_2["flow"], 0)
            timestep_now_3["flow"] = np.where(timestep_now_3["cleared_now"], timestep_now_3["flow"], 0)
            timestep_now_4["flow"] = np.where(timestep_now_4["cleared_now"], timestep_now_4["flow"], 0)


        # step 14: cleared fully booleans
        timestep_now_optimal["cleared_fully"] = attach_cleared_fully(master_data, timestep_now_optimal)
        timestep_now_1["cleared_fully"] = attach_cleared_fully(master_data, timestep_now_1)
        timestep_now_2["cleared_fully"] = attach_cleared_fully(master_data, timestep_now_2)
        timestep_now_3["cleared_fully"] = attach_cleared_fully(master_data, timestep_now_3)
        timestep_now_4["cleared_fully"] = attach_cleared_fully(master_data, timestep_now_4)


        # step 15: Step timestep df
        yearly_results_o[year] = timestep_now_optimal
        yearly_results_1[year] = timestep_now_1
        yearly_results_2[year] = timestep_now_2
        yearly_results_3[year] = timestep_now_3
        yearly_results_4[year] = timestep_now_4


        # step 16: assign previous timesteps
        prev_year_df_optimal = timestep_now_optimal.copy()
        prev_year_df_1 = timestep_now_1.copy()
        prev_year_df_2 = timestep_now_2.copy()
        prev_year_df_3 = timestep_now_3.copy()
        prev_year_df_4 = timestep_now_4.copy()


    # Post processing
    final_results = postprocess_yearly_results([yearly_results_o, yearly_results_1, yearly_results_2, yearly_results_3, yearly_results_4],comp_miu_df,follow_up_df,comp_only_df)

    #########################################################################################################
    return final_results, budgets
