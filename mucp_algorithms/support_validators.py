"""
Purpose: Validate functions for MUCP support data
Author: Kirodh Boodhraj
"""

import pandas as pd

# growth form validation
def validate_growth_form(growth_forms_list, clearing_norms_growth_form, species_growth_form):
    warnings, errors = [], []

    # get unique values
    growth_forms_set = set(growth_forms_list)
    clearing_norms_set = set(clearing_norms_growth_form)
    species_set = set(species_growth_form)

    # check clearing_norms_growth_form
    unknown_clearing = clearing_norms_set - growth_forms_set
    for val in unknown_clearing:
        errors.append(f"unknown clearing_norms_growth_form value '{val}' was not found in current growth forms options")

    # check species_growth_form
    unknown_species = species_set - growth_forms_set
    for val in unknown_species:
        errors.append(f"unknown species_growth_form value '{val}' was not found in current growth forms options")

    return {"warnings": warnings, "errors": errors}


# treatment methods validation
def validate_treatment_methods(treatment_methods_list, clearing_norms_treatment_methods):
    warnings, errors = [], []

    # get unique values
    treatment_methods_set = set(treatment_methods_list)
    clearing_norms_set = set(clearing_norms_treatment_methods)

    # check clearing_norms_growth_form
    unknown_clearing = clearing_norms_set - treatment_methods_set
    for val in unknown_clearing:
        errors.append(f"unknown clearing_norms_treatment_methods value '{val}' was not found in current treatment methods options")


    return {"warnings": warnings, "errors": errors}


# species validation
def validate_species(species_df, miu_linked_species, nbal_linked_species):
    warnings, errors = [], []

    # get unique values
    species_set = set(species_df["species_name"].tolist())
    miu_species_set = set(miu_linked_species)
    nbal_species_set = set(nbal_linked_species)

    # check clearing_norms_growth_form
    miu_unknown = miu_species_set - species_set
    nbal_unknown = nbal_species_set - species_set
    for val in miu_unknown:
        errors.append(f"unknown miu linked species value '{val}' was not found in current tree species options")
    for val in nbal_unknown:
        errors.append(f"unknown nbal linked species value '{val}' was not found in current tree species options")


    return {"warnings": warnings, "errors": errors}


# clearing norms validation
def validate_clearing_norms(clearing_norms_size_class_list, miu_size_class, nbal_size_class):
    warnings, errors = [], []

    # get unique values
    clearing_norms_size_class_set = set(clearing_norms_size_class_list)
    miu_size_class_set = set(miu_size_class)
    nbal_size_class_set = set(nbal_size_class)

    # Combine miu and nbal size classes
    all_size_classes = set(miu_size_class) | set(nbal_size_class)

    # Find which ones are missing in clearing norms
    missing_size_classes = all_size_classes - clearing_norms_size_class_set

    # Add to errors if any are missing
    for val in missing_size_classes:
        errors.append(
            f"Unknown size class value '{val}' was not found in current clearing norm size class options"
        )

    return {"warnings": warnings, "errors": errors}


# prioritization validation
def validate_prioritization_model(compartment_priorities_data, categories, headers_required):
    warnings, errors = [], []

    gdf = compartment_priorities_data.copy()

    # --- ERRORS ---
    compartment_header = headers_required[0]
    # 1. Required compartment_header column
    if compartment_header not in gdf.columns:
        errors.append(f"Missing required column '{compartment_header}' in input data.")

    if not categories:
        errors.append(f"No categories in planning case.")

    for category in categories:
        # col_name = category.name
        col_name = category["name"]

        # 2. Check if category.name is in gdf headers
        if col_name not in gdf.columns:
            errors.append(f"Missing required category column '{col_name}' in input data.")
            continue

        # 3. Category must have ranges/text values
        if category["type"] == "numeric":
            if not category.get("ranges"):
                errors.append(f"Category '{col_name}' has no numeric ranges defined.")
        elif category["type"] == "text":
            if not category.get("allowed"):
                errors.append(f"Category '{col_name}' has no allowed text values defined.")


    # --- WARNINGS ---
    for category in categories:
        col_name = category["name"]
        col_values = gdf[col_name]

        if category["weight"] < 0 or category["weight"] > 1:
            warnings.append(
                f"Category '{col_name}' has weight less than 0 or more than 1."
            )

        if category["type"] == "numeric":
            # 4. Non-numeric values -> warning + drop rows
            non_numeric_mask = pd.to_numeric(col_values, errors="coerce").isna()
            if non_numeric_mask.any():
                bad_rows = gdf.loc[non_numeric_mask, [compartment_header, col_name]]
                warnings.append(
                    f"Category '{col_name}' has non-numeric values. Dropping {len(bad_rows)} rows."
                )
                gdf = gdf.loc[~non_numeric_mask].copy()
                col_values = gdf[col_name]  # refresh

            # 5. Values outside defined bands
            valid_ranges = category["ranges"]

            def in_any_range(val):
                try:
                    v = float(val)
                    return any(low <= v <= high for (low, high, _priority) in valid_ranges)
                except Exception:
                    return False

            outside_mask = ~col_values.apply(in_any_range)
            if outside_mask.any():
                bad_rows = gdf.loc[outside_mask, [compartment_header, col_name]]
                warnings.append(
                    f"Category '{col_name}' has {len(bad_rows)} values outside all numeric bands. Dropping them."
                )
                gdf = gdf.loc[~outside_mask].copy()

            # 6. Priority check: ensure all priorities are numeric
            for (_low, _high, pr) in valid_ranges:
                if not isinstance(pr, (int, float)):
                    errors.append(
                        f"Category '{col_name}' has a non-numeric priority value: {pr}"
                    )

        elif category["type"] == "text":
            allowed_vals = {v["value"].lower() for v in category["allowed"]}

            invalid_mask = ~col_values.isin(allowed_vals)
            if invalid_mask.any():
                bad_rows = gdf.loc[invalid_mask, [compartment_header, col_name]]
                warnings.append(
                    f"Category '{col_name}' has {len(bad_rows)} values not in allowed text values. Dropping them."
                )

            # Check priority values are numeric
            for v in category["allowed"]:
                if not isinstance(v["priority"], (int, float)):
                    errors.append(
                        f"Category '{col_name}' has non-numeric priority for text value '{v['value']}': {v['priority']}"
                    )

    return {"warnings": warnings, "errors": errors}


# costing models validation
def validate_costing_models(costing_models_df, required_headers):
    warnings, errors = [], []

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
    costing_models_df = costing_models_df.drop_duplicates(subset=[costing_model_name_header])

    # --- Check required headers exist ---
    missing_headers = [h for h in required_headers if h not in costing_models_df.columns]
    if missing_headers:
        errors.append(f"Missing required columns: {', '.join(missing_headers)}")
        return {"warnings": warnings, "errors": errors}

    # --- Error checks (immediate return if found) ---
    for _, row in costing_models_df.iterrows():
        model_name = row[costing_model_name_header]

        if row[initial_team_size_header] < 0:
            errors.append(f"{model_name}: {initial_team_size_header} cannot be negative.")
        if row[initial_cost_per_day_header] < 0:
            errors.append(f"{model_name}: {initial_cost_per_day_header} cannot be negative.")
        if row[followup_team_size_header] < 0:
            errors.append(f"{model_name}: {followup_team_size_header} cannot be negative.")
        if row[followup_cost_per_day_header] < 0:
            errors.append(f"{model_name}: {followup_cost_per_day_header} cannot be negative.")
        if row[vehicle_cost_per_day_header] < 0:
            errors.append(f"{model_name}: {vehicle_cost_per_day_header} cannot be negative.")
        if row[fuel_cost_per_hour_header] < 0:
            errors.append(f"{model_name}: {fuel_cost_per_hour_header} cannot be negative.")
        if row[total_cost_per_day_header] < 0:
            errors.append(f"{model_name}: {total_cost_per_day_header} cannot be negative.")

    if errors:
        return {"warnings": warnings, "errors": errors}  # return immediately if errors

    # --- Warning checks ---
    for _, row in costing_models_df.iterrows():
        model_name = row[costing_model_name_header]

        if row[initial_team_size_header] == 0:
            warnings.append(f"{model_name}: {initial_team_size_header} is 0 – no initial team for clearing.")
        elif row[initial_team_size_header] > 50:
            warnings.append(f"{model_name}: {initial_team_size_header} is unusually high (>50).")

        if row[followup_team_size_header] == 0:
            warnings.append(f"{model_name}: {followup_team_size_header} is 0 – no follow-up team.")
        elif row[followup_team_size_header] > 50:
            warnings.append(f"{model_name}: {followup_team_size_header} is unusually high (>50).")


    return {"warnings": warnings, "errors": errors}


# planning variables validation
def validate_planning_variables(budget_plan_1, budget_plan_2, budget_plan_3, budget_plan_4, escalation_plan_1, escalation_plan_2, escalation_plan_3, escalation_plan_4, standard_working_day, standard_working_year_days, start_year, years_to_run, currency, save_results):
    warnings, errors = [], []

    # --- Budget checks ---
    budgets = [budget_plan_1, budget_plan_2, budget_plan_3, budget_plan_4]
    if any(b < 0 for b in budgets):
        errors.append("Budgets cannot be negative.")

    # --- Escalation checks ---
    escalations = [escalation_plan_1, escalation_plan_2, escalation_plan_3, escalation_plan_4]
    if any(es < 0 or es > 100 for es in escalations):
        errors.append("Escalation percentages must be between 0 and 100.")

    # --- Standard hours in working day ---
    if standard_working_day <= 0:
        errors.append("Standard working hours in day must be greater than 0 hours.")

    # --- Standard working days in year ---
    if standard_working_year_days <= 0:
        errors.append("Standard working days in year must be greater than 0 days.")

    # --- Standard working days in year ---
    if standard_working_year_days > 365:
        errors.append("Standard working days in year exceeds 1 full year. Values are between 1 and 365.")

    # --- Years to run ---
    if years_to_run <= 1:
        errors.append("Years to run must be greater than 0.")

    # --- Currency ---
    if not isinstance(currency, str) or not currency.strip():
        errors.append("Currency must be a non-empty string.")

    # --- Save results ---
    if not isinstance(save_results, bool):
        errors.append("Save results must be a boolean (True/False).")

    if errors:
        return {"warnings": warnings, "errors": errors}  # return immediately if errors

    # --- Budget checks ---
    # Increasing budget trend (not enforced, just warning)
    if not all(earlier <= later for earlier, later in zip(budgets, budgets[1:])):
        warnings.append("Budgets are not strictly increasing (expected Budget 1 < Budget 2 < Budget 3 < Budget 4).")

    # --- Standard working day ---
    if standard_working_day > 8:
        warnings.append("Standard working day exceeds 8 hours. This may be a lot for a team. Go easy on them!")

    # --- Standard working days in year ---
    if standard_working_year_days > 220:
        warnings.append("Standard working days in year exceeds 220 days! This may be a lot for a team. Go easy on them!")

    # --- Years to run ---
    if years_to_run > 20:
        warnings.append("Number of years to run exceeds 20. Simulation may take a long time.")

    # --- Start year ---
    if start_year < 1900 or start_year > 3000:
        warnings.append(f"Start year {start_year} is unusual. Check if correct.")

    return {"warnings": warnings, "errors": errors}

