"""
Purpose: Read support data functions for MUCP
Author: Kirodh Boodhraj

Read data from Excel file (see example Excel file in the examples for user input for the support data) or use it from
  the viewer passed as a list or df.

These functions will cleanup the data and validate them for you.

Note that you need to have opened the user uploaded files namely:
- miu shp,
- nbal shp,
- compartments shp,
- gis mapping shp,
- miu linked species Excel,
- nbal linked species Excel,
- compartment priorities csv,

"""

import pandas as pd
from .support_validators import validate_growth_form, validate_treatment_methods, validate_species, validate_clearing_norms, validate_prioritization_model, validate_costing_models, validate_planning_variables


# growth form
def read_growth_form(growth_forms_list, clearing_norms_growth_form, species_growth_form, validate = False):
    # the growth form is linked to the clearing_norms and the species
    # ensure lower
    growth_forms_list = [item.lower() for item in growth_forms_list if isinstance(item, str)]
    clearing_norms_growth_form = [item.lower() for item in clearing_norms_growth_form if isinstance(item, str)]
    species_growth_form = [item.lower() for item in species_growth_form if isinstance(item, str)]

    # validate
    if validate:
        return validate_growth_form(growth_forms_list, clearing_norms_growth_form, species_growth_form)

    # clean data

    # get unique values
    growth_forms_set = set(growth_forms_list)
    clearing_norms_set = set(clearing_norms_growth_form)
    species_set = set(species_growth_form)

    # Keep only growth forms that appear in either clearing_norms_set or species_set
    valid_growth_forms = growth_forms_set.intersection(clearing_norms_set.union(species_set))

    # Return as a sorted list (optional, for consistent order)
    return list(valid_growth_forms)


# treatment method
def read_treatment_methods(treatment_methods_list, clearing_norms_treatment_methods, validate = False):
    # ensure lower
    treatment_methods_list = [item.lower() for item in treatment_methods_list if isinstance(item, str)]
    clearing_norms_treatment_methods = [item.lower() for item in clearing_norms_treatment_methods if isinstance(item, str)]

    # validate
    if validate:
        return validate_treatment_methods(treatment_methods_list, clearing_norms_treatment_methods)

    # clean data

    # get unique values
    treatment_methods_set = set(treatment_methods_list)
    clearing_norms_set = set(clearing_norms_treatment_methods)

    # Keep only growth forms that appear in either clearing_norms_set or species_set
    valid_treatment_methods = treatment_methods_set.intersection(clearing_norms_set)

    # Return as a sorted list (optional, for consistent order)
    return list(valid_treatment_methods)


# species
def read_species(species_df, miu_linked_species, nbal_linked_species, validate = False):
    # ensure lower
    # Convert only object (string) columns to lowercase
    species_df = species_df.map(
        lambda x: x.lower() if isinstance(x, str) else x
    )
    miu_linked_species = [item.lower() for item in miu_linked_species if isinstance(item, str)]
    nbal_linked_species = [item.lower() for item in nbal_linked_species if isinstance(item, str)]

    # validate
    if validate:
        return validate_species(species_df, miu_linked_species, nbal_linked_species)

    # clean data

    # get unique values
    miu_linked_species_set = set(miu_linked_species)
    nbal_linked_species_set = set(nbal_linked_species)

    # union of both sets
    all_species_set = miu_linked_species_set | nbal_linked_species_set

    # filter species_df
    species_df = species_df[species_df["species_name"].isin(all_species_set)]

    # Strip strings in object fields
    for col in species_df.select_dtypes(include=["object"]).columns:
        species_df[col] = species_df[col].str.strip().str.lower()

    # Return as a sorted list (optional, for consistent order)
    return species_df

# herbicides, not used in tool at the moment
def read_herbicides(df): return df


# clearing norms
def read_clearing_norms(clearing_norms_df, miu_size_class, nbal_size_class, species_growth_forms, validate = False):
    # Convert only object (string) columns to lowercase
    clearing_norms_df = clearing_norms_df.map(
        lambda x: x.lower() if isinstance(x, str) else x
    )
    miu_size_class = [item.lower() for item in miu_size_class if isinstance(item, str)]
    nbal_size_class = [item.lower() for item in nbal_size_class if isinstance(item, str)]


    # validate
    if validate:
        return validate_clearing_norms(clearing_norms_df["size_class"], miu_size_class, nbal_size_class)


    # filter clearing_norm_df for growth forms in this case
    clearing_norms_df = clearing_norms_df[clearing_norms_df["growth_form"].isin(species_growth_forms)]

    # filter clearing_norm_df for size classes in this case
    # get unique values
    miu_size_class_set = set(miu_size_class)
    nbal_size_class_set = set(nbal_size_class)

    # Combine miu and nbal size classes
    all_size_classes = set(miu_size_class) | set(nbal_size_class)
    clearing_norms_df = clearing_norms_df[clearing_norms_df["size_class"].isin(all_size_classes)]

    # Strip strings in object fields
    for col in clearing_norms_df.select_dtypes(include=["object"]).columns:
        clearing_norms_df[col] = clearing_norms_df[col].str.strip().str.lower()

    return clearing_norms_df


# prioritization categories
def read_prioritization_categories(compartment_priorities_data, categories,validate = False, headers_required=["compt_id"]):

    # validate
    if validate:
        return validate_prioritization_model(compartment_priorities_data, categories, headers_required)

    gdf = compartment_priorities_data.copy()
    # compartment_header = headers_required[0]

    for category in categories:
        col_name = category["name"]
        if col_name not in gdf.columns:
            continue

        col_values = gdf[col_name]

        if category["type"] == "numeric":
            # Remove non-numeric
            non_numeric_mask = pd.to_numeric(col_values, errors="coerce").isna()
            gdf = gdf.loc[~non_numeric_mask]
            col_values = gdf[col_name]

            # Remove out-of-band
            def in_any_range(val):
                try:
                    v = float(val)
                    return any(low <= v <= high for (low, high, _priority) in category["ranges"])
                except Exception:
                    return False

            outside_mask = ~col_values.apply(in_any_range)
            gdf = gdf.loc[~outside_mask]

        elif category["type"] == "text":
            # Extract only the string values from allowed
            allowed_vals = {v["value"].lower() for v in category["allowed"]}

            invalid_mask = ~col_values.isin(allowed_vals)
            gdf = gdf.loc[~invalid_mask]

    # Keep only necessary columns
    category_cols = [c["name"] for c in categories if c["name"] in gdf.columns]
    gdf = gdf[headers_required + category_cols]

    # Drop rows with missing values in required headers
    gdf = gdf.dropna(subset=headers_required)

    # Strip strings in object fields
    for col in gdf.select_dtypes(include=["object"]).columns:
        gdf[col] = gdf[col].str.strip().str.lower()

    return gdf


# costing model
def read_costing_model(df: pd.DataFrame, required_headers: list = ["Costing Model Name","Initial Team Size","Initial Cost/Day", "Follow-up Team Size","Follow-up Cost/Day","Vehicle Cost/Day", "Fuel Cost/Hour","Maintenance Level","Cost/Day"],validate = False):
    # validate
    if validate:
        return validate_costing_models(df, required_headers)

    # Assign headers to variables for readability
    (
        costing_model_name_header,
        initial_team_size_header,
        initial_cost_per_day_header,
        followup_team_size_header,
        followup_cost_per_day_header,
        vehicle_cost_per_day_header,
        fuel_cost_per_hour_header,
        maintenance_level_header,
        total_cost_per_day_header,
    ) = required_headers

    # --- Deduplicate entries by costing model name ---
    df = df.drop_duplicates(subset=[costing_model_name_header])
    return df


# planning variables
def read_planning_variables(budget_plan_1, budget_plan_2, budget_plan_3, budget_plan_4, escalation_plan_1, escalation_plan_2, escalation_plan_3, escalation_plan_4, standard_working_day, standard_working_year_days, start_year, years_to_run, currency, save_results,validate = False):
    # validate
    if validate:
        return validate_planning_variables(budget_plan_1, budget_plan_2, budget_plan_3, budget_plan_4, escalation_plan_1, escalation_plan_2, escalation_plan_3, escalation_plan_4, standard_working_day, standard_working_year_days, start_year, years_to_run, currency, save_results)

    return float(budget_plan_1), float(budget_plan_2), float(budget_plan_3), float(budget_plan_4), float(escalation_plan_1), float(escalation_plan_2), float(escalation_plan_3), float(escalation_plan_4), float(standard_working_day), standard_working_year_days, start_year, years_to_run, currency, save_results  # placeholder
