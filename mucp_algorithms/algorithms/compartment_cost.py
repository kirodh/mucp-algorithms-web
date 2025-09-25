import pandas as pd
import numpy as np
from .compartment_priority import get_priorities
from .compartment_density import calculate_species_density
from .compartment_flow import calculate_flow
from .compartment_person_days import calculate_normal_person_days, calculate_adjusted_person_days



# -------------------------------
# Prioritization
# -------------------------------
def get_prioritization(prioritization_model_data, categories):
    return get_priorities(prioritization_model_data, categories)

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

def propagate_budgets_for_all_years(start_year: int, years_to_run: int, budget_1: float, budget_2: float, budget_3: float, budget_4: float, escalation_1: float, escalation_2: float, escalation_3: float, escalation_4: float):
    """
    Propagate budgets using simple interest per year.

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
            i = escalation_plans[plan] / 100.0  # convert % → decimal
            fv = pv * (1 + i * year_offset)  # simple interest
            results[year][plan] = fv
    return results


# -------------------------------
# Merge and split data frames for all data
# -------------------------------

def merge_gis_mapping_compartment_miu_nbal_df(compartment_data, miu_data ,nbal_data, gis_mapping_data):
    # step get all valid entries if they are in the miu, nbal and compartment df's
    # Build sets of valid IDs from the other dataframes
    valid_compt_ids = set(compartment_data["compt_id"].dropna().unique())
    valid_miu_ids = set(miu_data["miu_id"].dropna().unique())
    valid_nbal_ids = set(nbal_data["nbal_id"].dropna().unique())

    # Filter compartments → must always exist
    gis_mapping_data = gis_mapping_data[gis_mapping_data["compt_id"].isin(valid_compt_ids)]

    # Filter miu → only check rows where miu_id is not null
    mask_miu = gis_mapping_data["miu_id"].isna() | gis_mapping_data["miu_id"].isin(valid_miu_ids)
    gis_mapping_data = gis_mapping_data[mask_miu]

    # Filter nbal → only check rows where nbal_id is not null
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

def merge_prioritization_data(master_df, prioritization_df):
    return master_df.merge(
        prioritization_df,
        on="compt_id",
        how="left"  # keep all expanded_df rows, add costing data
    )

# -------------------------------
# Slope factor
# -------------------------------
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

        # If no matches → drop density filter
        if candidates.empty:
            mask = (
                (norms["growth_form"].str.lower() == row["growth_form"].lower()) &
                (norms["size_class"].str.lower() == row["age"].lower()) &
                (norms["terrain"].str.lower() == row["riparian_c"].lower())
            )
            candidates = norms[mask]

        # If still no matches → return None
        if candidates.empty:
            return None

        # If only one → take it
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
                (merged["Cost/Day"] * merged["person_days_normal"])
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
            (merged["Cost/Day"] * merged["person_days_normal"])
        )

    return cost_optimal

# clearing rules for compartment
def attach_cleared_now(
    master_df: pd.DataFrame,
    timestep_df: pd.DataFrame,
    year_budget: float,
) -> pd.DataFrame:
    # print("master_df.shape")
    # print(master_df.shape)
    # print("timestep_df.shape")
    # print(timestep_df.shape)

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


    # print("merged.shape")
    # print(merged.shape)

    # Return Series aligned to timestep_df
    return merged["cleared_now"].reset_index(drop=True)
    # return merged

# determine and add onto cleared fully
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
## directly inserted in costing loops, equation can be broadcasted into the arrays efficiently

# -------------------------------
# Flow Algorithms
# -------------------------------

# match the MAR to the compartment id in the master df
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


    # # Attach new column to the original df (preserve structure)
    # timestep_df_now = current_timestep.copy()
    # timestep_df_now["flow_optimal"] = flow_optimal
    #
    # return timestep_df_now

    return flow_optimal

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


    # # Attach new column to the original df (preserve structure)
    # timestep_df_now = current_timestep.copy()
    # timestep_df_now["flow_optimal"] = flow_optimal
    #
    # return timestep_df_now

    return flow_optimal



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



#### MAIN COSTING FUNCTION
def calculate_budgets(gis_mapping_data, miu_data, nbal_data, compartment_data, miu_linked_species_data, nbal_linked_species_data, compartment_priorities_data, growth_forms, treatment_methods, clearing_norms_df, species, costing_data, budget_plan_1, budget_plan_2, budget_plan_3, budget_plan_4, escalation_plan_1, escalation_plan_2, escalation_plan_3, escalation_plan_4, standard_working_day, working_year_days, start_year, years_to_run, currency, save_results, cost_plan_mappings, categories, prioritization_model_data):

    # step 1 fixed variables
    ## prioritization
    prioritization_df = get_prioritization(prioritization_model_data, categories)
    # print("prioritization")
    # print(prioritization_df.columns)
    # print(prioritization_df.iloc[0])


    # will flow be calculated?
    calculate_flow_boolean = "runoff" in compartment_priorities_data.columns


    # split the clearing norms into initial and followup
    clearing_norms_df_followup = clearing_norms_df[clearing_norms_df["process"] == "follow-up"].copy()
    clearing_norms_df_initial = clearing_norms_df[clearing_norms_df["process"] == "initial"].copy()
    # print(clearing_norms_df_followup)
    # print(clearing_norms_df_initial)

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


    # step 5 set the slope factor
    comp_miu_df["slope_factor"] = set_slope_factor(comp_miu_df)
    follow_up_df["slope_factor"] = set_slope_factor(follow_up_df)


    # step 6 insert tree species data
    comp_miu_df = merge_tree_species_data(comp_miu_df,miu_linked_species_data, nbal_linked_species_data,species,calculate_flow_boolean)
    follow_up_df = merge_tree_species_data(follow_up_df,miu_linked_species_data, nbal_linked_species_data,species,calculate_flow_boolean)


    # step 7 insert the cost model parameters
    comp_miu_df = merge_cost_model_data(comp_miu_df,costing_data)
    follow_up_df = merge_cost_model_data(follow_up_df,costing_data)


    # step 8 merge the priority into the data
    comp_miu_df = merge_prioritization_data(comp_miu_df, prioritization_df)
    follow_up_df = merge_prioritization_data(follow_up_df, prioritization_df)

    # step 9 determine treatment method
    comp_miu_df = treatment_selection(comp_miu_df, clearing_norms_df_initial)
    follow_up_df = treatment_selection(follow_up_df, clearing_norms_df_followup)

    # step 10 add on the fuel cost per miu
    # Count how many miu_id per compt_id
    miu_counts_initial = comp_miu_df.groupby("compt_id")["miu_id"].transform("nunique")
    miu_counts_followup = follow_up_df.groupby("compt_id")["miu_id"].transform("nunique")
    # Apply cost formula, its a fixed cost
    ## (Fuel cost per Hour * Drive Time) / number of miu's per compartment / 10
    comp_miu_df["fuel_cost_per_miu"] = (comp_miu_df["Fuel Cost/Hour"] * comp_miu_df["drive_time"] / (miu_counts_initial * 10))
    follow_up_df["fuel_cost_per_miu"] = (follow_up_df["Fuel Cost/Hour"] * follow_up_df["drive_time"] / (miu_counts_followup * 10))

    # step 10 drop unneccesary columns
    unnecessary_columns_to_drop = ["stage", "area_miu", "costing", "area_ha", "slope"]
    comp_miu_df = comp_miu_df.drop(columns=unnecessary_columns_to_drop, errors="ignore")
    follow_up_df = follow_up_df.drop(columns=unnecessary_columns_to_drop, errors="ignore")



    # step 11 propagate budgets
    budgets = propagate_budgets_for_all_years(start_year, years_to_run, budget_plan_1, budget_plan_2, budget_plan_3, budget_plan_4, escalation_plan_1, escalation_plan_2, escalation_plan_3, escalation_plan_4)
    # print("budgets kirodh")
    # print(budgets)
    # step 12 insert flow mean annual runoff if its to be calculated
    comp_miu_df = attach_runoff_to_compartment(compartment_priorities_data, comp_miu_df)
    follow_up_df = attach_runoff_to_compartment(compartment_priorities_data, follow_up_df)






    # print(prioritization_df)
    # print(budgets)

    # step 13 assign unique ids to link back to the dfs
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

    ###
    # step 13 deal with year 1, i.e. initial stuff

    #########################################################################################################
    # step 13 deal with year years

    # budget dictionaries
    yearly_results_o = {} # optimal
    yearly_results_1 = {}
    yearly_results_2 = {}
    yearly_results_3 = {}
    yearly_results_4 = {}

    # --- Timestep 0 (initial year) ---
    initial_year = start_year

    ### INITIAL
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
    timestep_df_0_miu_initial["density_optimal"] = calculate_species_density(comp_miu_df["idenscode"], comp_miu_df["initial_reduction"])

    # 4. flow
    if calculate_flow_boolean:
        timestep_df_0_miu_initial["flow_optimal"] = attach_flow_initial_timestep(comp_miu_df, timestep_df_0_miu_initial, follow_up=False)



    ### FOLLOW UP
    # construct the initial time step follow-up calculations
    timestep_df_0_nbal_followup = pd.DataFrame({
        "link_back_id": follow_up_df["link_back_id"],
    })

    # ppd
    timestep_df_0_nbal_followup["person_days_factor"] = attach_person_day_factor(follow_up_df, follow_up_df["idenscode"], clearing_norms_df_followup)
    # normal person days
    timestep_df_0_nbal_followup["person_days_normal"] = calculate_normal_person_days(timestep_df_0_nbal_followup["person_days_factor"], follow_up_df["area"])
    # adjusted person days:
    timestep_df_0_nbal_followup["person_days"] = calculate_adjusted_person_days(timestep_df_0_nbal_followup["person_days_normal"], follow_up_df["walk_time"], follow_up_df["drive_time"], follow_up_df["slope_factor"], standard_working_day)

    # 2. costs
    # cost_optimal
    timestep_df_0_nbal_followup["cost_optimal"] = attach_cost(follow_up_df,timestep_df_0_nbal_followup)

    # densities
    # density_optimal is for assuming all densities were done
    # density is the compartments that werent cleared are densified, ad the cleared ones are reduced
    timestep_df_0_nbal_followup["density_optimal"] = calculate_species_density(follow_up_df["idenscode"], follow_up_df["initial_reduction"])

    # flow
    if calculate_flow_boolean:
        timestep_df_0_nbal_followup["flow_optimal"] = attach_flow_initial_timestep(follow_up_df, timestep_df_0_nbal_followup)



    ### next phase
    # Combine timestep DataFrames, for each budget
    timestep_df_0_optimal = pd.concat(
        [timestep_df_0_miu_initial, timestep_df_0_nbal_followup],
        ignore_index=True  # resets index, but keeps link_back_id intact
    )

    # Remove "_optimal" from column names
    timestep_df_0_optimal = timestep_df_0_optimal.rename(
        columns=lambda c: c.replace("_optimal", "")
    ).copy()

    # Create copies for timestep_df_0_1 through timestep_df_0_4
    timestep_df_0_1 = timestep_df_0_optimal.copy()
    timestep_df_0_2 = timestep_df_0_optimal.copy()
    timestep_df_0_3 = timestep_df_0_optimal.copy()
    timestep_df_0_4 = timestep_df_0_optimal.copy()


    # combine the initial master df into one, as all will be follow up from now on after calculating the first time step
    master_data = pd.concat(
        # [comp_only_df, comp_miu_df, follow_up_df], # remember to add back on the compartment only entries
        [comp_miu_df, follow_up_df],
        ignore_index=True  # resets index, but keeps link_back_id intact
    )


    # Add a cleared fully column, initial cleared fully previously variable
    timestep_df_0_optimal["cleared_fully_previously"] = False
    timestep_df_0_1["cleared_fully_previously"] = False
    timestep_df_0_2["cleared_fully_previously"] = False
    timestep_df_0_3["cleared_fully_previously"] = False
    timestep_df_0_4["cleared_fully_previously"] = False


    # add the prioritizations on:
    timestep_df_0_optimal["priority"] = attach_prioritization(master_data, timestep_df_0_1)
    timestep_df_0_1["priority"] = attach_prioritization(master_data, timestep_df_0_1)
    timestep_df_0_2["priority"] = attach_prioritization(master_data, timestep_df_0_2)
    timestep_df_0_3["priority"] = attach_prioritization(master_data, timestep_df_0_3)
    timestep_df_0_4["priority"] = attach_prioritization(master_data, timestep_df_0_4)


    # do the clearing and the final values for person days, density etc for each budget
    # clearing now booleans
    timestep_df_0_optimal["cleared_now"] = True # for optimal all is cleared
    timestep_df_0_1["cleared_now"] = attach_cleared_now(master_data, timestep_df_0_1, budgets[np.int64(initial_year)]['plan_1'])
    timestep_df_0_2["cleared_now"] = attach_cleared_now(master_data, timestep_df_0_2, budgets[np.int64(initial_year)]['plan_2'])
    timestep_df_0_3["cleared_now"] = attach_cleared_now(master_data, timestep_df_0_3, budgets[np.int64(initial_year)]['plan_3'])
    timestep_df_0_4["cleared_now"] = attach_cleared_now(master_data, timestep_df_0_4, budgets[np.int64(initial_year)]['plan_4'])

    # density per budget
    # Pick factor depending on cleared_now
    density_factors_initial_1 = np.where(timestep_df_0_1["cleared_now"], master_data["initial_reduction"], master_data["densification"])
    density_factors_initial_2 = np.where(timestep_df_0_2["cleared_now"], master_data["initial_reduction"], master_data["densification"])
    density_factors_initial_3 = np.where(timestep_df_0_3["cleared_now"], master_data["initial_reduction"], master_data["densification"])
    density_factors_initial_4 = np.where(timestep_df_0_4["cleared_now"], master_data["initial_reduction"], master_data["densification"])
    # Compute density
    timestep_df_0_1["density"] = calculate_species_density(master_data["idenscode"], density_factors_initial_1)
    timestep_df_0_1["density"] = calculate_species_density(master_data["idenscode"], density_factors_initial_1)
    timestep_df_0_2["density"] = calculate_species_density(master_data["idenscode"], density_factors_initial_2)
    timestep_df_0_3["density"] = calculate_species_density(master_data["idenscode"], density_factors_initial_3)
    timestep_df_0_4["density"] = calculate_species_density(master_data["idenscode"], density_factors_initial_4)

    # person days per budget
    # 2.5 person days adjustment after clearing
    # adjusted person days:
    timestep_df_0_1["person_days"] = np.where(timestep_df_0_1["cleared_now"], timestep_df_0_1["person_days"], 0)
    timestep_df_0_2["person_days"] = np.where(timestep_df_0_2["cleared_now"], timestep_df_0_2["person_days"], 0)
    timestep_df_0_3["person_days"] = np.where(timestep_df_0_3["cleared_now"], timestep_df_0_3["person_days"], 0)
    timestep_df_0_4["person_days"] = np.where(timestep_df_0_4["cleared_now"], timestep_df_0_4["person_days"], 0)

    # cost per budget
    timestep_df_0_1["cost"] = np.where(timestep_df_0_1["cleared_now"], timestep_df_0_1["cost"], 0)
    timestep_df_0_2["cost"] = np.where(timestep_df_0_2["cleared_now"], timestep_df_0_2["cost"], 0)
    timestep_df_0_3["cost"] = np.where(timestep_df_0_3["cleared_now"], timestep_df_0_3["cost"], 0)
    timestep_df_0_4["cost"] = np.where(timestep_df_0_4["cleared_now"], timestep_df_0_4["cost"], 0)

    # flow per budget
    if calculate_flow_boolean:
        timestep_df_0_1["flow"] = np.where(timestep_df_0_1["cleared_now"], timestep_df_0_1["flow"], 0)
        timestep_df_0_2["flow"] = np.where(timestep_df_0_2["cleared_now"], timestep_df_0_2["flow"], 0)
        timestep_df_0_3["flow"] = np.where(timestep_df_0_3["cleared_now"], timestep_df_0_3["flow"], 0)
        timestep_df_0_4["flow"] = np.where(timestep_df_0_4["cleared_now"], timestep_df_0_4["flow"], 0)

    # cleared fully booleans
    timestep_df_0_optimal["cleared_fully"] = attach_cleared_fully(master_data,timestep_df_0_optimal)
    timestep_df_0_1["cleared_fully"] = attach_cleared_fully(master_data,timestep_df_0_1)
    timestep_df_0_2["cleared_fully"] = attach_cleared_fully(master_data,timestep_df_0_2)
    timestep_df_0_3["cleared_fully"] = attach_cleared_fully(master_data,timestep_df_0_3)
    timestep_df_0_4["cleared_fully"] = attach_cleared_fully(master_data,timestep_df_0_4)



    # Save as first timestep df , cut out the right columns and rename
    yearly_results_o[initial_year] = timestep_df_0_optimal
    yearly_results_1[initial_year] = timestep_df_0_1
    yearly_results_2[initial_year] = timestep_df_0_2
    yearly_results_3[initial_year] = timestep_df_0_3
    yearly_results_4[initial_year] = timestep_df_0_4


    # --- Timestep 1+ (follow-up years) ---
    prev_year_df_optimal = timestep_df_0_optimal.copy()
    prev_year_df_1 = timestep_df_0_1.copy()
    prev_year_df_2 = timestep_df_0_2.copy()
    prev_year_df_3 = timestep_df_0_3.copy()
    prev_year_df_4 = timestep_df_0_4.copy()

    for year in range(start_year + 1, start_year + years_to_run):
        # print(budgets[year]["plan_1"])
        # print(budgets[year]["plan_2"])
        # print(budgets[year]["plan_3"])
        # print(budgets[year]["plan_4"])

        # DONE step 1: create stub of new timestep
        # Create timestep_now with selected and renamed columns
        timestep_now_optimal = timestep_df_0_optimal.loc[:, ["link_back_id", "priority", "cleared_fully"]].rename(columns={"cleared_fully": "cleared_fully_previously"}).copy()
        timestep_now_1 = timestep_df_0_1.loc[:, ["link_back_id", "priority", "cleared_fully"]].rename(columns={"cleared_fully": "cleared_fully_previously"}).copy()
        timestep_now_2 = timestep_df_0_2.loc[:, ["link_back_id", "priority", "cleared_fully"]].rename(columns={"cleared_fully": "cleared_fully_previously"}).copy()
        timestep_now_3 = timestep_df_0_3.loc[:, ["link_back_id", "priority", "cleared_fully"]].rename(columns={"cleared_fully": "cleared_fully_previously"}).copy()
        timestep_now_4 = timestep_df_0_4.loc[:, ["link_back_id", "priority", "cleared_fully"]].rename(columns={"cleared_fully": "cleared_fully_previously"}).copy()


        # align the previous timestep data to the master_data, this is to ensure when you slice data they correspond to the same rows
        prev_year_df_optimal = (prev_year_df_optimal.set_index("link_back_id").reindex(master_data["link_back_id"]).reset_index())
        prev_year_df_1 = (prev_year_df_1.set_index("link_back_id").reindex(master_data["link_back_id"]).reset_index())
        prev_year_df_2 = (prev_year_df_2.set_index("link_back_id").reindex(master_data["link_back_id"]).reset_index())
        prev_year_df_3 = (prev_year_df_3.set_index("link_back_id").reindex(master_data["link_back_id"]).reset_index())
        prev_year_df_4 = (prev_year_df_4.set_index("link_back_id").reindex(master_data["link_back_id"]).reset_index())


        # ppd
        # calculate the person days factor
        timestep_now_optimal["person_days_factor"] = attach_person_day_factor(master_data, prev_year_df_optimal["density"], clearing_norms_df_followup)
        timestep_now_1["person_days_factor"] = attach_person_day_factor(master_data, prev_year_df_1["density"], clearing_norms_df_followup)
        timestep_now_2["person_days_factor"] = attach_person_day_factor(master_data, prev_year_df_2["density"], clearing_norms_df_followup)
        timestep_now_3["person_days_factor"] = attach_person_day_factor(master_data, prev_year_df_3["density"], clearing_norms_df_followup)
        timestep_now_4["person_days_factor"] = attach_person_day_factor(master_data, prev_year_df_4["density"], clearing_norms_df_followup)

        # normal person days
        timestep_now_optimal["person_days_normal"] = calculate_normal_person_days(timestep_now_optimal["person_days_factor"], master_data["area"])
        timestep_now_1["person_days_normal"] = calculate_normal_person_days(timestep_now_1["person_days_factor"], master_data["area"])
        timestep_now_2["person_days_normal"] = calculate_normal_person_days(timestep_now_2["person_days_factor"], master_data["area"])
        timestep_now_3["person_days_normal"] = calculate_normal_person_days(timestep_now_3["person_days_factor"], master_data["area"])
        timestep_now_4["person_days_normal"] = calculate_normal_person_days(timestep_now_4["person_days_factor"], master_data["area"])

        # adjusted person days:
        timestep_now_optimal["person_days"] = calculate_adjusted_person_days(timestep_now_optimal["person_days_normal"], master_data["walk_time"], master_data["drive_time"],master_data["slope_factor"], standard_working_day)
        timestep_now_1["person_days"] = calculate_adjusted_person_days(timestep_now_1["person_days_normal"], master_data["walk_time"], master_data["drive_time"],master_data["slope_factor"], standard_working_day)
        timestep_now_2["person_days"] = calculate_adjusted_person_days(timestep_now_2["person_days_normal"], master_data["walk_time"], master_data["drive_time"],master_data["slope_factor"], standard_working_day)
        timestep_now_3["person_days"] = calculate_adjusted_person_days(timestep_now_3["person_days_normal"], master_data["walk_time"], master_data["drive_time"],master_data["slope_factor"], standard_working_day)
        timestep_now_4["person_days"] = calculate_adjusted_person_days(timestep_now_4["person_days_normal"], master_data["walk_time"], master_data["drive_time"],master_data["slope_factor"], standard_working_day)

        # 2. costs
        # cost_optimal
        timestep_now_optimal["cost"] = attach_cost(master_data, timestep_now_optimal)
        timestep_now_1["cost"] = attach_cost(master_data, timestep_now_1)
        timestep_now_2["cost"] = attach_cost(master_data, timestep_now_2)
        timestep_now_3["cost"] = attach_cost(master_data, timestep_now_3)
        timestep_now_4["cost"] = attach_cost(master_data, timestep_now_4)


        # assign old densities to the timestep, so it can be used in the costing and then assign new densities over it
        timestep_now_optimal["density"] = timestep_now_optimal["link_back_id"].map(prev_year_df_optimal.set_index("link_back_id")["density"])
        timestep_now_1["density"] = timestep_now_1["link_back_id"].map(prev_year_df_1.set_index("link_back_id")["density"])
        timestep_now_2["density"] = timestep_now_2["link_back_id"].map(prev_year_df_2.set_index("link_back_id")["density"])
        timestep_now_3["density"] = timestep_now_3["link_back_id"].map(prev_year_df_3.set_index("link_back_id")["density"])
        timestep_now_4["density"] = timestep_now_4["link_back_id"].map(prev_year_df_4.set_index("link_back_id")["density"])

        # do the clearing and the final values for person days, density etc for each budget
        # clearing now booleans
        timestep_now_optimal["cleared_now"] = True  # for optimal all is cleared
        timestep_now_1["cleared_now"] = attach_cleared_now(master_data, timestep_now_1,budgets[np.int64(year)]['plan_1'])
        timestep_now_2["cleared_now"] = attach_cleared_now(master_data, timestep_now_2,budgets[np.int64(year)]['plan_2'])
        timestep_now_3["cleared_now"] = attach_cleared_now(master_data, timestep_now_3,budgets[np.int64(year)]['plan_3'])
        timestep_now_4["cleared_now"] = attach_cleared_now(master_data, timestep_now_4,budgets[np.int64(year)]['plan_4'])


        # density per budget (assign over the old densities)
        # Pick factor depending on cleared_now
        density_factors_now_optimal = np.where(timestep_now_optimal["cleared_now"], master_data["initial_reduction"],master_data["densification"])
        density_factors_now_1 = np.where(timestep_now_1["cleared_now"], master_data["initial_reduction"],master_data["densification"])
        density_factors_now_2 = np.where(timestep_now_2["cleared_now"], master_data["initial_reduction"],master_data["densification"])
        density_factors_now_3 = np.where(timestep_now_3["cleared_now"], master_data["initial_reduction"],master_data["densification"])
        density_factors_now_4 = np.where(timestep_now_4["cleared_now"], master_data["initial_reduction"],master_data["densification"])
        # Compute density
        timestep_now_optimal["density"] = calculate_species_density(prev_year_df_optimal["density"], density_factors_now_optimal)
        timestep_now_1["density"] = calculate_species_density(prev_year_df_1["density"], density_factors_now_1)
        timestep_now_2["density"] = calculate_species_density(prev_year_df_2["density"], density_factors_now_2)
        timestep_now_3["density"] = calculate_species_density(prev_year_df_3["density"], density_factors_now_3)
        timestep_now_4["density"] = calculate_species_density(prev_year_df_4["density"], density_factors_now_4)

        # calculate flow:
        if calculate_flow_boolean:
            timestep_now_optimal["flow"] = attach_flow(master_data, timestep_now_optimal)
            timestep_now_1["flow"] = attach_flow(master_data, timestep_now_1)
            timestep_now_2["flow"] = attach_flow(master_data, timestep_now_2)
            timestep_now_3["flow"] = attach_flow(master_data, timestep_now_3)
            timestep_now_4["flow"] = attach_flow(master_data, timestep_now_4)

        # person days per budget
        # 2.5 person days adjustment after clearing
        # adjusted person days:
        timestep_now_optimal["person_days"] = np.where(timestep_now_optimal["cleared_now"], timestep_now_optimal["person_days"], 0)
        timestep_now_1["person_days"] = np.where(timestep_now_1["cleared_now"], timestep_now_1["person_days"], 0)
        timestep_now_2["person_days"] = np.where(timestep_now_2["cleared_now"], timestep_now_2["person_days"], 0)
        timestep_now_3["person_days"] = np.where(timestep_now_3["cleared_now"], timestep_now_3["person_days"], 0)
        timestep_now_4["person_days"] = np.where(timestep_now_4["cleared_now"], timestep_now_4["person_days"], 0)

        # cost per budget
        timestep_now_optimal["cost"] = np.where(timestep_now_optimal["cleared_now"], timestep_now_optimal["cost"], 0)
        timestep_now_1["cost"] = np.where(timestep_now_1["cleared_now"], timestep_now_1["cost"], 0)
        timestep_now_2["cost"] = np.where(timestep_now_2["cleared_now"], timestep_now_2["cost"], 0)
        timestep_now_3["cost"] = np.where(timestep_now_3["cleared_now"], timestep_now_3["cost"], 0)
        timestep_now_4["cost"] = np.where(timestep_now_4["cleared_now"], timestep_now_4["cost"], 0)



        # flow per budget
        if calculate_flow_boolean:
            timestep_now_optimal["flow"] = np.where(timestep_now_optimal["cleared_now"], timestep_now_optimal["flow"], 0)
            timestep_now_1["flow"] = np.where(timestep_now_1["cleared_now"], timestep_now_1["flow"], 0)
            timestep_now_2["flow"] = np.where(timestep_now_2["cleared_now"], timestep_now_2["flow"], 0)
            timestep_now_3["flow"] = np.where(timestep_now_3["cleared_now"], timestep_now_3["flow"], 0)
            timestep_now_4["flow"] = np.where(timestep_now_4["cleared_now"], timestep_now_4["flow"], 0)

        # cleared fully booleans
        timestep_now_optimal["cleared_fully"] = attach_cleared_fully(master_data, timestep_now_optimal)
        timestep_now_1["cleared_fully"] = attach_cleared_fully(master_data, timestep_now_1)
        timestep_now_2["cleared_fully"] = attach_cleared_fully(master_data, timestep_now_2)
        timestep_now_3["cleared_fully"] = attach_cleared_fully(master_data, timestep_now_3)
        timestep_now_4["cleared_fully"] = attach_cleared_fully(master_data, timestep_now_4)


        # Step timestep df
        yearly_results_o[year] = timestep_now_optimal
        yearly_results_1[year] = timestep_now_1
        yearly_results_2[year] = timestep_now_2
        yearly_results_3[year] = timestep_now_3
        yearly_results_4[year] = timestep_now_4

        # assign previous timesteps
        prev_year_df_optimal = timestep_now_optimal.copy()
        prev_year_df_1 = timestep_now_1.copy()
        prev_year_df_2 = timestep_now_2.copy()
        prev_year_df_3 = timestep_now_3.copy()
        prev_year_df_4 = timestep_now_4.copy()

    # print("BEFORE")
    # print(yearly_results_o)
    # print(yearly_results_1)
    # print(yearly_results_2)
    # print(yearly_results_3)
    # print(yearly_results_4)
    # Post processing
    final_results = postprocess_yearly_results([yearly_results_o, yearly_results_1, yearly_results_2, yearly_results_3, yearly_results_4],comp_miu_df,follow_up_df,comp_only_df)
    # print("AFTER")
    # print(final_results[0])
    # print(final_results[1])
    # print(final_results[2])
    # print(final_results[3])
    # print(final_results[4][2026].columns)

    year = 2026
    costs_per_budget = {}

    # loop through all 5 budgets
    for i in range(5):
        df = final_results[i][year]
        costs_per_budget[f"budget_{i + 1}"] = df["cost"].reset_index(drop=True)

    # # combine into one DataFrame for comparison
    # comparison_df = pd.concat(costs_per_budget, axis=1)

    # print(comparison_df)

    #########################################################################################################
    # Final master dataframe
    # print("types")
    # print(type(final_results))
    # print(type(final_results[0][2025]))
    # print(type(budgets[2025]))
    return final_results, budgets

# def calculate_budgets(self):
#     """
#     Main budgeting loop: calculate monthly compartment budgets over multiple years.
#
#     Returns:
#         final_budgets_yearly: list of DataFrames with yearly results for each budget option.
#     """
#     # -----------------------------
#     # 1. Validate basic input parameters
#     # -----------------------------
#     number_of_years = self.mucp_input_file.planning_years.item()
#     working_day_hours = self.mucp_input_file.operations_working_day_hours.item()
#     working_year_days = self.mucp_input_file.operations_working_year_days.item()
#
#     if number_of_years > 50 or working_day_hours > 16 or working_year_days > 300:
#         return []
#
#     # -----------------------------
#     # 2. Prepare initial setup dataframe
#     # -----------------------------
#     miu_data = self.miu_linked_species_file.data.copy()
#     nbal_data = self.nbal_linked_species_file.data.copy()
#     gis_data = self.gis_file.data.copy()
#
#     # Merge NBAL and MIU data with GIS mapping
#     nbal_initial = pd.merge(gis_data[gis_data['nbal_id'].notnull()], nbal_data, on='nbal_id', how='left')
#     miu_initial = pd.merge(gis_data[gis_data['nbal_id'].isnull()], miu_data, on='miu_id', how='left')
#     initial_setup = pd.concat([nbal_initial, miu_initial], ignore_index=True)
#     initial_setup = initial_setup.rename(columns={'idenscode': 'density'})
#
#     # -----------------------------
#     # 3. Populate constants
#     # -----------------------------
#     initial_setup['initial'] = initial_setup['nbal_id'].apply(self.is_initial_treatment)
#     initial_setup['riparian'] = initial_setup['miu_id'].apply(self.get_riparian)
#
#     # Vectorized compartment info
#     initial_setup[['slope', 'walk_time', 'drive_time', 'costing_code', 'growth_condition']] = \
#         pd.DataFrame(initial_setup['compt_id'].map(self.get_compartment_info).tolist(), index=initial_setup.index)
#
#     initial_setup[['initial_density_reduction', 'followup_density_reduction', 'densification', 'treatment_frequency',
#                    'growth_form']] = \
#         pd.DataFrame(initial_setup['species'].map(self.get_species_constants).tolist(), index=initial_setup.index)
#
#     initial_setup['flow_reduction_factor'] = initial_setup.apply(
#         lambda row: self.get_flow_reduction_factor(row['compt_id'], row['age'], row['species'],
#                                                    row['growth_condition']), axis=1
#     )
#
#     initial_setup[
#         ['cost_plan', 'initial_team_size', 'initial_cost_per_day', 'followup_team_size', 'followup_cost_per_day',
#          'vehicle_cost_per_day', 'fuel_cost_per_hour', 'daily_cost', 'maintenance_level']] = \
#         pd.DataFrame(initial_setup['costing_code'].map(self.get_costing_model).tolist(), index=initial_setup.index)
#
#     # -----------------------------
#     # 4. Remove error rows and adjust density reductions
#     # -----------------------------
#     required_cols = ['compt_id', 'area', 'costing_code', 'cost_plan', 'density', 'initial', 'growth_form', 'walk_time',
#                      'drive_time', 'area']
#     error_rows_df = initial_setup[initial_setup[required_cols].isna().any(axis=1)].copy()
#     initial_setup.dropna(subset=required_cols, inplace=True)
#
#     initial_setup[['initial_density_reduction', 'followup_density_reduction']] *= -1
#
#     # -----------------------------
#     # 5. Additional setup: slope factor, prioritization, MAR
#     # -----------------------------
#     initial_setup['slope_factor'] = initial_setup['slope'].map(self.get_slope_factor)
#     prioritization = self.get_prioritization()
#     initial_setup['prioritization'] = initial_setup["compt_id"].map(
#         lambda c: self.set_prioritization(c, prioritization))
#
#     mar_cols = ["mean_annual_runoff", "mean annual runoff", "mar", "annual runoff", "mean runoff", "mean_runoff",
#                 "annual_runoff", "runoff"]
#     mar_columns = [col for col in self.compartment_priority_file.data.columns if col in mar_cols]
#     initial_setup['mean_annual_runoff'] = initial_setup['compt_id'].map(
#         lambda c: self.set_mean_annual_runoff_per_compartment(c, self.compartment_priority_file.data, mar_columns)
#     )
#
#     # -----------------------------
#     # 6. Ensure non-zero team sizes/costs
#     # -----------------------------
#     initial_setup['initial_team_size'].replace(0, 1, inplace=True)
#     initial_setup['initial_cost_per_day'].replace(0, 1, inplace=True)
#     initial_setup['followup_team_size'].replace(0, 1, inplace=True)
#     initial_setup['followup_cost_per_day'].replace(0, 1, inplace=True)
#
#     # -----------------------------
#     # 7. Map categorical values to integers for speed
#     # -----------------------------
#     initial_setup.iloc[:, 18] = initial_setup.iloc[:, 18].map(self.mapping_growth_form)
#     initial_setup.iloc[:, 6] = initial_setup.iloc[:, 6].map(self.mapping_age)
#     initial_setup.iloc[:, 8] = initial_setup.iloc[:, 8].map(self.mapping_riparian)
#
#     # -----------------------------
#     # 8. Initialize timestep 0
#     # -----------------------------
#     time_step_0 = pd.DataFrame({
#         'density': initial_setup['density'],
#         'person_days_normal': 0,
#         'person_days': 0,
#         'flow': 0,
#         'cost': 0,
#         'is_cleared': False,
#         'initial': initial_setup['initial']
#     })
#
#     # -----------------------------
#     # 9. Compute yearly funds and treatment steps
#     # -----------------------------
#     financial_funds = self.get_yearly_funds(number_of_years)
#     financial_values = financial_funds.values
#     treatment_frequencies = initial_setup['treatment_frequency'].unique().tolist()
#     if 12 not in treatment_frequencies:
#         treatment_frequencies.append(12)
#     total_steps = number_of_years * 12 + 1
#
#     # Create treatment mask matrix for timesteps
#     row_indices = np.arange(total_steps)
#     step_mask = pd.DataFrame({f: row_indices % f for f in treatment_frequencies}).mask(lambda x: x.ne(0), np.nan)
#     step_mask = step_mask.fillna({col: int(col) for col in step_mask.columns}).values
#     step_mask[0, :] = np.nan
#
#     # -----------------------------
#     # 10. Precompute fuel costs per compartment
#     # -----------------------------
#     fuel_cost_all = self.fuel_formula(initial_setup[['fuel_cost_per_hour', 'drive_time']]).rename('fuel_cost')
#     fuel_cost_df = pd.concat([initial_setup['compt_id'], fuel_cost_all], axis=1)
#     fuel_cost_per_compartment = self.sort_fuel_compartment_costing(fuel_cost_df)
#
#     # -----------------------------
#     # 11. Initialize final budgets array
#     # -----------------------------
#     n_rows = initial_setup.shape[0]
#     final_budgets = np.full((financial_funds.shape[1], total_steps + 1, n_rows, 7), np.nan)
#     final_budgets[:, 0, :, :] = time_step_0.values
#
#     # -----------------------------
#     # 12. Loop over budgets and timesteps
#     # -----------------------------
#     for b_idx, budget_name in enumerate(financial_funds.columns):
#         for t_idx, timestep in enumerate(step_mask):
#             # Set yearly budget
#             if t_idx % 12 == 0:
#                 current_budget = financial_values[int(t_idx / 12), b_idx]
#                 print(budget_name, int(t_idx / 12), current_budget)
#
#             # Copy previous timestep if no treatment
#             if np.isnan(timestep).all():
#                 final_budgets[b_idx, t_idx + 1] = final_budgets[b_idx, t_idx]
#                 continue
#
#             # Prepare temp array with previous timestep
#             temp_arr = np.full((n_rows, 9), np.nan)
#             prev_step = final_budgets[b_idx, t_idx]
#             temp_arr[:, 0:7] = prev_step
#
#             # Determine active treatment mask
#             active_treatments = timestep[~np.isnan(timestep)]
#             treatment_mask = np.isin(initial_setup["treatment_frequency"].values, active_treatments).reshape(-1, 1)
#
#             # -----------------------------
#             # 12a. Calculate density
#             # -----------------------------
#             temp_arr[:, [-2, -1]] = self.calculate_density(
#                 temp_arr[:, [0, 5, 6]],
#                 initial_setup[['densification', 'initial_density_reduction', 'followup_density_reduction']].values
#             )
#
#             # -----------------------------
#             # 12b. Calculate person days
#             # -----------------------------
#             temp_arr[:, [1, 2]] = self.calculate_person_days(
#                 temp_arr[:, [0, 5, 6]],
#                 initial_setup[
#                     ['growth_form', 'age', 'riparian', 'walk_time', 'drive_time', 'area', 'slope_factor']].values,
#                 working_day_hours
#             )
#
#             # -----------------------------
#             # 12c. Calculate costing
#             # -----------------------------
#             temp_arr[:, [0, 2, 4, 5, 6]], current_budget, is_costed = self.calculate_costing(
#                 temp_arr[:, [0, 1, 2, -4, -3, -2, -1]],
#                 initial_setup[
#                     ['initial_team_size', 'initial_cost_per_day', 'followup_team_size', 'followup_cost_per_day',
#                      'vehicle_cost_per_day', 'fuel_cost_per_hour', 'daily_cost', 'maintenance_level',
#                      'prioritization', 'drive_time']].values,
#                 treatment_mask,
#                 current_budget,
#                 fuel_cost_per_compartment
#             )
#
#             # -----------------------------
#             # 12d. Calculate flow
#             # -----------------------------
#             temp_arr[:, 3] = self.calculate_flow(
#                 temp_arr[:, [0, 6]],
#                 initial_setup[['area', 'riparian', 'flow_reduction_factor', 'mean_annual_runoff',
#                                'densification', 'initial_density_reduction', 'followup_density_reduction']].values,
#                 is_costed
#             )
#
#             final_budgets[b_idx, t_idx + 1] = temp_arr[:, 0:7]
#
#     # -----------------------------
#     # 13. Condense monthly budgets into yearly results
#     # -----------------------------
#     final_budgets_yearly = [
#         [
#             pd.DataFrame(
#                 np.concatenate((initial_setup[['compt_id', 'miu_id', 'nbal_id', 'area']].values,
#                                 j[:, [0, 2, 3, 4]]), axis=1),
#                 columns=['output_compartment_id', 'output_miu_id', 'output_nbal_id', 'output_area',
#                          'output_density', 'output_person_days', 'output_flow', 'output_cost']
#             ).assign(
#                 output_budget_option='Optimal' if i == 4 else f'Budget_{i + 1}',
#                 output_year=index + 1
#             ) for index, j in enumerate(final_budgets[i, ::12])
#         ] for i in range(final_budgets.shape[0])
#     ]
#
#     return final_budgets_yearly

