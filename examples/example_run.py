"""
Purpose:
Example case files:
-miu shp
-nbal shp
-compartment shp
-gis mapping shp

-miu linked species excel
-nbal linked species excel

-compartment prioritization categories csv

-user mucp support data input excel

How to run a case. Open files here and then link to the actual algorithms.
"""
import pandas as pd
import geopandas as gpd
from mucp_algorithms import data_reader,support_data_reader
from mucp_algorithms.algorithms.compartment_cost import calculate_budgets


# helper functions:
def is_data_valid(validation_result: dict) -> bool:
    """Check if validation result has no errors or warnings."""
    return not validation_result.get("errors")


def build_categories(weights_df, bands_df, prioritization_model: str):
    categories = []

    # --- Normalize column names ---
    weights_df.columns = weights_df.columns.str.strip().str.lower()
    # print(weights_df.columns)
    bands_df.columns = bands_df.columns.str.strip().str.lower()
    # print(bands_df.columns)

    # --- Find row in weights_df that matches prioritization_model ---
    match_row = weights_df[
        weights_df["prioritization model"].str.strip().str.lower() == prioritization_model.lower()
    ]
    if match_row.empty:
        raise ValueError(f"No row found for prioritization model '{prioritization_model}'")

    # Only one row should match
    row = match_row.iloc[0]

    # --- Iterate over each category column in the weights row ---
    for col_name, weight in row.items():
        # print(col_name,weight)
        if col_name == "prioritization model":
            continue  # skip the header col

        # collect the corresponding bands slice
        # pandas assigns a .1 .2 etc.to the same name columns
        band_slices = bands_df[[col_name,col_name+".1",col_name+".2"]].dropna(how="all").iloc[1:]
        # print(band_slices)
        # print(len(band_slices))
        if len(band_slices) == 0:
            # print(f"SKIPPING THIS ONE {col_name}")
            continue  # no bands → skip

        # check if the range high column is all nan, then it is numeric
        if band_slices[col_name+".1"].isna().all():
            type = "text"
        else:
            type = "numeric"


        # Build category structure
        cat_struct = {"name": col_name, "type":type , "weight": float(weight) if not pd.isna(weight) else 0.0}

        if type == "text":
            # allowed = band_slices[col_name].dropna().str.lower().unique().tolist()
            # cat_struct["allowed"] = allowed
            values = band_slices[col_name].dropna().str.strip().str.lower().tolist()
            priorities_text = band_slices[col_name + ".2"].dropna().astype(float).tolist()

            # Pair up values with priorities
            allowed = [
                {"value": v, "priority": p}
                for v, p in zip(values, priorities_text)
            ]
            cat_struct["allowed"] = allowed
        else: # its numeric
            lows = band_slices[col_name].dropna().astype(float)
            highs = band_slices[col_name+".1"].dropna().astype(float)
            priority_numeric = band_slices[col_name+".2"].dropna().astype(float)
            ranges = list(zip(lows, highs, priority_numeric))
            cat_struct["ranges"] = ranges


        categories.append(cat_struct)

    return categories


def plot_me(costing, budgets):
    pass



def run_mucp():
    #################
    # READ INPUT DATA
    #################

    ## filenames
    gis_mapping_path = "example_case_files/H60B_GIS_mapping_tm19.shp"
    gis_mapping_validations = data_reader.read_gis_mapping_shapefile(gis_mapping_path, validate=True, headers_required = ["nbal_id", "miu_id", "compt_id","area"], headers_other = ["geometry"])
    if is_data_valid(gis_mapping_validations):
        print("Validated: gis mapping")
        gis_mapping_data = data_reader.read_gis_mapping_shapefile(gis_mapping_path, validate=False, headers_required=["nbal_id", "miu_id", "compt_id", "area"], headers_other=["geometry"])
    else:
        gis_mapping_data = gpd.GeoDataFrame(columns=["nbal_id", "miu_id", "compt_id", "area", "geometry"], geometry="geometry", crs="EPSG:4326")


    miu_path = "example_case_files/H60B_MIU_tm19.shp"
    miu_validations = data_reader.read_miu_shapefile(miu_path, gis_mapping_data["miu_id"].tolist(), validate=True, headers_required=["miu_id", "area", "riparian_c"], headers_other=["geometry"])
    if is_data_valid(miu_validations):
        print("Validated: miu shp")
        miu_data = data_reader.read_miu_shapefile(miu_path, gis_mapping_data["miu_id"].tolist(), validate=False, headers_required=["miu_id", "area", "riparian_c"], headers_other=["geometry"])
    else:
        miu_data = gpd.GeoDataFrame(columns=["miu_id", "area", "riparian_c", "geometry"], geometry="geometry", crs="EPSG:4326")


    nbal_path = "example_case_files/H60B_NBAL_tm19.shp"
    nbal_validations = data_reader.read_nbal_shapefile(nbal_path, gis_mapping_data["nbal_id"].tolist(), validate=True, headers_required=["nbal_id", "area", "stage"], headers_other=["geometry", "contractid", "first_date", "last_date"])
    if is_data_valid(nbal_validations):
        print("Validated: nbal shp")
        nbal_data = data_reader.read_nbal_shapefile(nbal_path, gis_mapping_data["nbal_id"].tolist(), validate=False, headers_required=["nbal_id", "area", "stage"], headers_other=["geometry", "contractid", "first_date", "last_date"])
    else:
        nbal_data = gpd.GeoDataFrame(columns=["nbal_id", "area", "stage", "geometry"], geometry="geometry", crs="EPSG:4326")

    compartment_path = "example_case_files/H60B_compartments_tm19.shp"
    compartment_validations = data_reader.read_compartment_shapefile(compartment_path, gis_mapping_data["compt_id"].tolist(), validate=True, headers_required=["compt_id", "area_ha", "slope", "walk_time", "drive_time", "costing", "grow_con"], headers_other=["geometry", "terrain"])
    if is_data_valid(compartment_validations):
        print("Validated: compartment shp")
        compartment_data = data_reader.read_compartment_shapefile(compartment_path, gis_mapping_data["compt_id"].tolist(), validate=False, headers_required=["compt_id", "area_ha", "slope", "walk_time", "drive_time", "costing", "grow_con"], headers_other=["geometry", "terrain"])
    else:
        compartment_data = gpd.GeoDataFrame(columns=["compt_id", "area_ha", "slope", "walk_time", "drive_time", "costing", "grow_con", "geometry"], geometry="geometry", crs="EPSG:4326")

    nbal_linked_species_path = "example_case_files/H60B_NBAL_linked_species.xlsx"
    nbal_linked_species_validations = data_reader.read_nbal_linked_species_excel(nbal_linked_species_path, validate=True, headers_required=["nbal_id", "species", "idenscode", "age"])
    if is_data_valid(nbal_linked_species_validations):
        print("Validated: mbal linked excel")
        nbal_linked_species_data = data_reader.read_nbal_linked_species_excel(nbal_linked_species_path, validate=False, headers_required=["nbal_id", "species", "idenscode", "age"])
    else:
        nbal_linked_species_data = pd.DataFrame(columns=["nbal_id", "species", "idenscode", "age"])

    miu_linked_species_path = "example_case_files/H60B_MIU_linked_species.xlsx"
    miu_linked_species_validations = data_reader.read_miu_linked_species_excel(miu_linked_species_path, validate=True, headers_required=["miu_id", "species", "idenscode", "age"])
    if is_data_valid(miu_linked_species_validations):
        print("Validated: miu linked excel")
        miu_linked_species_data = data_reader.read_miu_linked_species_excel(miu_linked_species_path, validate=False, headers_required=["miu_id", "species", "idenscode", "age"])
    else:
        miu_linked_species_data = pd.DataFrame(columns=["miu_id", "species", "idenscode", "age"])

    compartment_priorities_path = "example_case_files/H60B_compartments_priorities.csv"
    compartment_priorities_validations = data_reader.read_compartment_priorities_csv(compartment_priorities_path, validate=True, headers_required=["compt_id"])
    if is_data_valid(compartment_priorities_validations):
        print("Validated: compartment priorities csv")
        compartment_priorities_data = data_reader.read_compartment_priorities_csv(compartment_priorities_path, validate=False, headers_required=["compt_id"])
    else:
        compartment_priorities_data = pd.DataFrame(columns=["compt_id"])


    ## support data
    mucp_input_filename = "example_case_files/MUCP_support_data_user_input.xlsx"

    # support Excel spreadsheet names
    support_spreadsheet_names = ["Planning_Budgets",
                                 "Support_Clearing_Norms",
                                 "Support_Species",
                                 "Support_Prioritization_Model",
                                 "Support_Costing",
                                 "Support_Herbicides",
                                 "Operations",
                                 "Support_Growth_Form",
                                 "Support_Treat_Methods",
                                 "Priority_Categories",
                                 "Planning"]

    mucp_input_dfs = [pd.read_excel(mucp_input_filename, sheet_name=sheet) for sheet in support_spreadsheet_names]
    # # set all the variables here per spreadsheet:
    budget_plan_1 = mucp_input_dfs[0]["Budget Amount"][0]
    budget_plan_2 = mucp_input_dfs[0]["Budget Amount"][1]
    budget_plan_3 = mucp_input_dfs[0]["Budget Amount"][2]
    budget_plan_4 = mucp_input_dfs[0]["Budget Amount"][3]
    escalation_plan_1 = mucp_input_dfs[0]["% escalation"][0]
    escalation_plan_2 = mucp_input_dfs[0]["% escalation"][1]
    escalation_plan_3 = mucp_input_dfs[0]["% escalation"][2]
    escalation_plan_4 = mucp_input_dfs[0]["% escalation"][3]
    standard_working_day = mucp_input_dfs[6]["working_day_hours"][0]
    working_year_days = mucp_input_dfs[6]["working_year_days"][0]
    prioritization_model = mucp_input_dfs[10]["Prioritization Model"][0]
    start_year = mucp_input_dfs[10]["start_year"][0]
    years_to_run = mucp_input_dfs[10]["Years"][0]
    currency = mucp_input_dfs[10]["currency"][0]
    save_results = True


    # clearing norms
    clearing_norms = mucp_input_dfs[1]
    clearing_norms.columns = clearing_norms.columns.str.lower()
    for col in clearing_norms.select_dtypes(include=["object"]).columns:
        clearing_norms[col] = clearing_norms[col].str.lower()


    # species
    species = mucp_input_dfs[2]
    species.columns = species.columns.str.lower()
    for col in species.select_dtypes(include=["object"]).columns:
        species[col] = species[col].str.lower()


    ## validate and get support data
    # species validate and data
    species_validations = support_data_reader.read_species(species, miu_linked_species_data["species"].tolist(), nbal_linked_species_data["species"].tolist(), validate=True)
    if is_data_valid(species_validations):
        species = support_data_reader.read_species(species, miu_linked_species_data["species"].tolist(), nbal_linked_species_data["species"].tolist(), validate=False)
        print("SUPPORT Validated: species")


    # clearing norms validate and data
    clearing_norms_validations = support_data_reader.read_clearing_norms(clearing_norms, miu_linked_species_data["age"].tolist(), nbal_linked_species_data["age"].tolist(), species["growth_form"].tolist(), validate=True)
    if is_data_valid(clearing_norms_validations):
        clearing_norms_df = support_data_reader.read_clearing_norms(clearing_norms, miu_linked_species_data["age"].tolist(), nbal_linked_species_data["age"].tolist(), species["growth_form"].tolist(), validate=False)
        print("SUPPORT Validated: clearing norms")

    # growth forms
    growth_forms = mucp_input_dfs[7]["Growth Form"].dropna().astype(str).str.strip().str.lower().tolist()
    growth_forms_validations = support_data_reader.read_growth_form(growth_forms, clearing_norms["growth_form"].tolist(), species["growth_form"].tolist(), validate=True)
    if is_data_valid(growth_forms_validations):
        print("SUPPORT Validated: growth forms")


    # treatment methods
    treatment_methods = mucp_input_dfs[8]["Treatment Method"].dropna().astype(str).str.strip().str.lower().tolist()
    # # treatment method validate (use list (treatment_method) above for data)
    treatment_methods_validations = support_data_reader.read_treatment_methods(treatment_methods, clearing_norms["treatment_method"].tolist(), validate=True)
    if is_data_valid(treatment_methods_validations):
        print("SUPPORT Validated: treatment methods")


    # categories
    categories = build_categories(mucp_input_dfs[3], mucp_input_dfs[9], prioritization_model)

    prioritization_model_validations = support_data_reader.read_prioritization_categories(compartment_priorities_data, categories, validate=True, headers_required=["compt_id"])
    # print(prioritization_model_validations)
    if is_data_valid(prioritization_model_validations):
        prioritization_model_data = support_data_reader.read_prioritization_categories(compartment_priorities_data, categories, validate=False, headers_required=["compt_id"])
        print("SUPPORT Validated: prioritization categories")

    # cost code linkage
    cost_plan_mappings = {int(row["Code"]): row["Cost Plan"] for _, row in mucp_input_dfs[0].dropna().iterrows()}
    costing_before_validation = mucp_input_dfs[4]

    costing_validations = support_data_reader.read_costing_model(costing_before_validation, required_headers=["Costing Model Name", "Initial Team Size", "Initial Cost/Day", "Follow-up Team Size", "Follow-up Cost/Day", "Vehicle Cost/Day", "Fuel Cost/Hour", "Maintenance Level", "Cost/Day"], validate=True)
    if is_data_valid(costing_validations):
        costing_data = support_data_reader.read_costing_model(costing_before_validation, required_headers=["Costing Model Name", "Initial Team Size", "Initial Cost/Day", "Follow-up Team Size", "Follow-up Cost/Day", "Vehicle Cost/Day", "Fuel Cost/Hour", "Maintenance Level", "Cost/Day"], validate=False)
        print("SUPPORT Validated: costing")

    planning_validations = support_data_reader.read_planning_variables(budget_plan_1, budget_plan_2,
                                                                       budget_plan_3, budget_plan_4,
                                                                       escalation_plan_1,
                                                                       escalation_plan_2,
                                                                       escalation_plan_3,
                                                                       escalation_plan_4,
                                                                       standard_working_day,
                                                                       working_year_days,
                                                                       start_year, years_to_run,
                                                                       currency, save_results,
                                                                       validate=True)
    if is_data_valid(planning_validations):
        planning_budget_plan_1, planning_budget_plan_2, planning_budget_plan_3, planning_budget_plan_4, planning_escalation_plan_1, planning_escalation_plan_2, planning_escalation_plan_3, planning_escalation_plan_4, planning_standard_working_day, planning_working_year_days, planning_start_year, planning_years_to_run, planning_currency, planning_save_results = support_data_reader.read_planning_variables(
            budget_plan_1, budget_plan_2, budget_plan_3, budget_plan_4,
            escalation_plan_1, escalation_plan_2, escalation_plan_3,
            escalation_plan_4, standard_working_day, working_year_days, start_year, years_to_run,
            currency, save_results, validate=False)
        print("SUPPORT Validated: planning variables")


    #################
    # RUN SIMULATIONS
    #################

    # print("gis_mapping_data")
    # # print(gis_mapping_data)
    # print(gis_mapping_data.columns)
    # print(gis_mapping_data.iloc[0])
    # print('miu_data')
    # # print(miu_data)
    # print(miu_data.columns)
    # print(miu_data.iloc[0])
    # print("nbal_data")
    # # print(nbal_data)
    # print(nbal_data.columns)
    # print(nbal_data.iloc[0])
    # print("compartment_data")
    # # print(compartment_data)
    # print(compartment_data.columns)
    # print(compartment_data.iloc[0])
    # print("miu_linked_species_data")
    # # print(miu_linked_species_data)
    # print(miu_linked_species_data.columns)
    # print(miu_linked_species_data.iloc[0])
    # print("nbal_linked_species_data")
    # # print(nbal_linked_species_data)
    # print(nbal_linked_species_data.columns)
    # print(nbal_linked_species_data.iloc[0])
    # print("compartment_priorities_data")
    # print(compartment_priorities_data)
    # print(compartment_priorities_data.columns)
    # print(compartment_priorities_data.iloc[0])
    # print("growth_forms")
    # print(growth_forms)
    # print("treatment_methods")
    # print(treatment_methods)
    # print("clearing_norms_df")
    # print(clearing_norms_df)
    # print(clearing_norms_df.columns)
    # print(clearing_norms_df.iloc[0])
    # print("species")
    # print(species)
    # print(species.columns)
    # print(species.iloc[0])
    # print("costing_data")
    # print(costing_data)
    # print(costing_data.columns)
    # print(costing_data.iloc[0])
    # print("budget_plan_1")
    # print(budget_plan_1)
    # print("budget_plan_2")
    # print(budget_plan_2)
    # print("budget_plan_3")
    # print(budget_plan_3)
    # print("budget_plan_4")
    # print(budget_plan_4)
    # print("escalation_plan_1")
    # print(escalation_plan_1)
    # print("escalation_plan_2")
    # print(escalation_plan_2)
    # print("escalation_plan_3")
    # print(escalation_plan_3)
    # print("escalation_plan_4")
    # print(escalation_plan_4)
    # print("standard_working_day")
    # print(standard_working_day)
    # print("working_year_days")
    # print(working_year_days)
    # print("prioritization_model") # not for viewer, viewer selects this already
    # print(prioritization_model)
    # print("start_year")
    # print(start_year)
    # print("years_to_run")
    # print(years_to_run)
    # print("currency")
    # print(currency)
    # print("save_results")
    # print(save_results)
    # print("cost_plan_mappings")
    # print(cost_plan_mappings)
    # print("categories")
    # print(categories)
    # print("prioritization_model_data")
    # print(prioritization_model_data)
    # print(prioritization_model_data.columns)
    # print(prioritization_model_data.iloc[0])


    # costing
    costing, budgets = calculate_budgets(gis_mapping_data, miu_data, nbal_data, compartment_data, miu_linked_species_data, nbal_linked_species_data, compartment_priorities_data, growth_forms, treatment_methods, clearing_norms_df, species, costing_data, budget_plan_1, budget_plan_2, budget_plan_3, budget_plan_4, escalation_plan_1, escalation_plan_2, escalation_plan_3, escalation_plan_4, standard_working_day, working_year_days, start_year, years_to_run, currency, save_results, cost_plan_mappings, categories, prioritization_model_data)

    # temporary plotting
    plot_me(costing, budgets)

    return costing, budgets


if __name__ == '__main__':
    run_mucp()





