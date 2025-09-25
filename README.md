can you have one person for R100 a year
what would it take in csir for this t be streamlined?
which template is for above
can the templates accommodate international stuff
slide 3, seems like drugs and contrckt killing is legal...


can you please add this algoritms for the costing one please:
## Prioritization calculation
    def get_prioritization(self):
        # create prioritization object
        prioritization = Priority()
        # calculate priority vales for compartments
        prioritization = prioritization.get_priorities(self.mucp_input_file.planning_prioritization_plan[0],self.mucp_input_file.support_prioritization_model,self.mucp_input_file.support_priority_category_data,self.compartment_priority_file.compartment_priority_data)
        # set prioritization values to the compartment names
        return prioritization
        # return Priority(self.compartment_priority_data,self.mucp_input_data,self.gis_data,self.nbal_shp_data,self.miu_shp_data,self.compartment_shp_data,self.nbal_linked_species_data,self.miu_linked_species_data).get_priorities()

    # set prioritization to initial data
    def set_prioritization(self,row_data,prioritization):
        return prioritization[prioritization["compt_id"] == row_data]["priority"].values[0]

    # Function to get the budget for a given year
    def get_current_year_budget(self, cost_model, year):
        cost_model_name = list(cost_model['Costing Model Name'])[0]

        initialBudget = list(self.mucp_input_file.planning_budget_data[
                                 self.mucp_input_file.planning_budget_data['Cost Plan'] == cost_model_name][
                                 'Budget Amount'])[0]
        percent_escalation = list(self.mucp_input_file.planning_budget_data[
                                      self.mucp_input_file.planning_budget_data['Cost Plan'] == cost_model_name][
                                      '% escalation'])[0]

        return (initialBudget * (1 + (percent_escalation / 100)) ** year)

    # calculating the linked species cost
    def calculate_miu_linked_species_cost(self, pd_adjusted, pd_normal, vehicle_cost_perday, initial_team_cost_per_day,initial_team_size, user_defined_cost_perday):
        # exceptions
        initial_team_size = 1 if (initial_team_size == 0) else initial_team_size
        initial_team_cost_per_day = 1 if (initial_team_cost_per_day == 0) else initial_team_cost_per_day

        cost_per_linked_species_entry = (pd_adjusted * (initial_team_cost_per_day / initial_team_size)) + (
                    vehicle_cost_perday * pd_normal) + (user_defined_cost_perday * pd_normal)
        return cost_per_linked_species_entry

    # Function to find initial value for given COMP ID
    def is_initial_treatment(self,nbal_id):
        if pd.isna(nbal_id):  # Check if comp_value is NaN
            return True # Initial treatment

        # Find row in compartment_data dataframe based on COMP value
        nbal_row = self.nbal_shp_file.data[self.nbal_shp_file.data['nbal_id'] == nbal_id]

        if not nbal_row.empty:
            # Get initial value from compartment_data dataframe
            initial_value = nbal_row['stage'].iloc[0]

            # Return True if initial value is 'Initial', otherwise False
            return 'initial' in initial_value.lower()
        else:
            print("ERROR: There is no matching nbal id in the nbal shapefile!",nbal_id)
            return True

    def get_species_constants(self,species_name):
        if pd.isna(species_name):  # Check if comp_value is NaN
            return pd.Series({'initial_density_reduction':np.nan,'followup_density_reduction':np.nan,'densification':np.nan,'treatment_frequency':np.nan,'flow_reduction_factor':np.nan,'growth_form':np.nan})

        # Find row in compartment_data dataframe based on COMP value
        species_row = self.mucp_input_file.support_species_data[self.mucp_input_file.support_species_data['species name'] == species_name]
        if not species_row.empty:
            return pd.Series({
                'initial_density_reduction':species_row['initial reduction'].item(),
                'followup_density_reduction':species_row['follow-up reduction'].item(),
                'densification':species_row['densification'].item(),
                'treatment_frequency':species_row['treatment frequency'].item(),
                'growth_form':species_row['growth form'].item(),
                # 'flow_reduction_factor':species_row['flow optimal'].item(),
            })
        else:
            return pd.Series({'initial_density_reduction':np.nan,'followup_density_reduction':np.nan,'densification':np.nan,'treatment_frequency':np.nan,'flow_reduction_factor':np.nan, 'growth_form':np.nan})

    def get_compartment_info(self,compartment_id):
        if pd.isna(compartment_id):  # Check if comp_value is NaN
            print('ERROR: Compartment ID does not exist!!')
            return pd.Series({'slope':np.nan,'walk_time':np.nan,'drive_time':np.nan,'costing_code':np.nan,'growth_condition':np.nan})

        # Find row in compartment_data dataframe based on COMP value
        compartment_row = self.compartment_shp_file.data[self.compartment_shp_file.data['compt_id'] == compartment_id]
        if not compartment_row.empty:
            return pd.Series({  # ['', '', '', 'costing','growth_condition']
                'slope':compartment_row['slope'].item(),
                'walk_time':compartment_row['walk_time'].item(),
                'drive_time':compartment_row['drive_time'].item(),
                'costing_code':compartment_row['costing'].item(),
                'growth_condition':compartment_row['grow_con'].item(),
            })
        else:
            return pd.Series({'slope':np.nan,'walk_time':np.nan,'drive_time':np.nan,'costing_code':np.nan,'growth_condition':np.nan})

    def get_flow_reduction_factor(self,compt_id,age,species,growth_condition):
        if pd.isna(compt_id) or pd.isna(age) or pd.isna(species) or pd.isna(growth_condition):  # Check if value is NaN
            print('ERROR: Compartment ID or age or species or growth condition information does not exist!!')
            return np.nan

        # Find row in compartment_data dataframe based on COMP value
        species_row = self.mucp_input_file.support_species_data[self.mucp_input_file.support_species_data['species name'] == species]
        if not species_row.empty:
            if age == "young":
                return species_row['flow young'].item()
            elif age == "seedling":
                return species_row['flow seedling'].item()
            elif age == "coppice":
                return species_row['flow coppice'].item()
            elif age in ["mature","adult","mixed"]:
                if growth_condition == "sub-optimal":
                    return species_row['flow sub optimal'].item()
                else: # flow assumed to be optimal
                    return species_row['flow optimal'].item()
            else:
                print("ERROR: Invalid or missing growth form!! Treating as mature.")
                if growth_condition == "sub-optimal":
                    return species_row['flow sub optimal'].item()
                else:  # flow assumed to be optimal
                    return species_row['flow optimal'].item()
        else:
            return np.nan

    # todo set mean annual runoff per compartment to initial data
    def set_mean_annual_runoff_per_compartment(self, row_data, MAR, MAR_column_name):
        return MAR[MAR["compt_id"] == row_data][MAR_column_name].values[0][0]

    # TODO get flow
    def calculate_flow(self,flow_matrix,species_matrix,was_costed):
        flows = np.concatenate((flow_matrix, species_matrix,np.full((len(flow_matrix[:,0]),3),np.nan)), axis=1)

        # assign was_costed:
        flows[:,-3] = was_costed

        # determine density factor
        ## if not was_costed, use densification factor, otherwise  if initial, use initial reduction factor else followup, use followup reduction factor
        flows[:, -2] = np.where(flows[:,-3],np.where(flows[:,1],flows[:,7],flows[:,8]),flows[:,6])
        # calculate flow:
        flows[:,-1] = Flow().calculate_flow(flows[:,5],flows[:,2],flows[:,0],flows[:,-2],flows[:,4],flows[:,3])
        return flows[:,-1]

    def get_riparian(self,miu_id):
        if pd.isna(miu_id):  # Check if NaN
            return "landscape" # default value

        # Find row in compartment_data dataframe based on COMP value
        riparian = self.miu_shp_file.data[self.miu_shp_file.data['miu_id'] == miu_id]
        if not riparian.empty:
            if riparian['riparian_c'].item() == "l":
                return "landscape"
            else:
                return "riparian"
        else:
            return "landscape" # default value

    # get costing
    def get_costing_model(self, costing_code):
        empty_return = pd.Series({
                'cost_plan':np.nan,
                'initial_team_size':np.nan,
                'initial_cost_per_day':np.nan,
                'followup_team_size':np.nan,
                'followup_cost_per_day':np.nan,
                'vehicle_cost_per_day':np.nan,
                'fuel_cost_per_hour':np.nan,
                'daily_cost':np.nan,
                'maintenance_level':np.nan,
            })

        if pd.isna(costing_code):  # Check if NaN
            print('ERROR: Costing Code does not exist!!')
            return empty_return

        # Find row in budgeting models dataframe based on cost code
        budget_row = self.mucp_input_file.planning_budget_data[self.mucp_input_file.planning_budget_data['code'] == costing_code]
        if not budget_row.empty:
            # find the row in the costing model dataframe
            costing_model = self.mucp_input_file.support_costing_data[self.mucp_input_file.support_costing_data['costing model name'] == budget_row['cost plan'].item()]
            if not costing_model.empty:
                return pd.Series({
                    'cost_plan':costing_model['costing model name'].item(),
                    'initial_team_size':costing_model['initial team size'].item(),
                    'initial_cost_per_day':costing_model['initial cost/day'].item(),
                    'followup_team_size':costing_model['follow-up team size'].item(),
                    'followup_cost_per_day':costing_model['follow-up cost/day'].item(),
                    'vehicle_cost_per_day':costing_model['vehicle cost/day'].item(),
                    'fuel_cost_per_hour':costing_model['fuel cost/hour'].item(),
                    'daily_cost':costing_model['cost/day'].item(),
                    'maintenance_level':costing_model['maintenance level'].item(),
                })
            else:
                print("ERROR: Costing model not defined according to costing code!!!")
                return empty_return
        else: # no cost code
            print("ERROR: Cost code does not exist!!")
            return empty_return

    # calculate financial yearly funds available
    def get_yearly_funds(self,number_of_years):
        interest_rates = [
            self.mucp_input_file.planning_budget_data[self.mucp_input_file.planning_budget_data['budget'] == "plan 1"]["% escalation"].item(),
            self.mucp_input_file.planning_budget_data[self.mucp_input_file.planning_budget_data['budget'] == "plan 2"]["% escalation"].item(),     self.mucp_input_file.planning_budget_data[self.mucp_input_file.planning_budget_data['budget'] == "plan 3"]["% escalation"].item(),
self.mucp_input_file.planning_budget_data[self.mucp_input_file.planning_budget_data['budget'] == "plan 4"]["% escalation"].item()
        ]
        interest_factors = np.array(interest_rates) * .01
        funds_array = np.zeros((number_of_years + 1,4))
        funds_array[0] = [
            self.mucp_input_file.planning_budget_data[self.mucp_input_file.planning_budget_data['budget'] == "plan 1"]["budget amount"].item(),
            self.mucp_input_file.planning_budget_data[self.mucp_input_file.planning_budget_data['budget'] == "plan 2"]["budget amount"].item(),
            self.mucp_input_file.planning_budget_data[self.mucp_input_file.planning_budget_data['budget'] == "plan 3"]["budget amount"].item(),
            self.mucp_input_file.planning_budget_data[self.mucp_input_file.planning_budget_data['budget'] == "plan 4"]["budget amount"].item()
        ]

        for i in range(1, len(funds_array[:,0])):
            funds_array[i] = funds_array[i - 1] * (1 + interest_factors) # simple interest

        funds_df = pd.DataFrame(funds_array,columns=['budget_1','budget_2','budget_3','budget_4'])
        # add a column for optimal budgets, use a really large number so that it simulates an unlimited budget
        funds_df["optimal"] = 1e30
        return funds_df

    # slope factor
    def get_slope_factor(self,slope):
        ## slope:
        if slope >= 51:
            s_factor = 2
        elif slope >= 41:
            s_factor = 1.8
        elif slope >= 31:
            s_factor = 1.6
        elif slope >= 21:
            s_factor = 1.4
        elif slope >= 11:
            s_factor = 1.2
        elif slope >= 0:
            s_factor = 1
        else:  # shouldnt have this slope
            print("ERROR: Slope is invalid. Defaulting to slope 1")
            s_factor = 1
        return s_factor

    # calculate density
    def calculate_density(self,previous_density_matrix,densification_factor_matrix):


        ## returns an array of size (?,2) with columns densification and density reduction

        densities = np.concatenate((previous_density_matrix, densification_factor_matrix,np.full((len(densification_factor_matrix[:,0]),3),np.nan)), axis=1)

        # populate if initial or followup reduction
        densities[:,-3] = np.where(densities[:, 2], densities[:, 4], densities[:, 5])

        # densify:
        densities[:,-2] = Density().calculate_species_density(densities[:,0], densities[:,3])
        # density reduction:
        densities[:,-1] = Density().calculate_species_density(densities[:,0], densities[:,-3]) # or index 6

        return densities[:,[-2,-1]]

    def treatment_selection(self,norms):
        # selection process of the norms according to treatment method:

        index_adult = self.mapping_age.get("adult")
        index_landscape = self.mapping_riparian.get("landscape")
        index_riparian = self.mapping_riparian.get("riparian")
        index_felling = self.mapping_treatment_method.get("felling")
        index_ring_bark = self.mapping_treatment_method.get("ring bark")
        index_bark_strip = self.mapping_treatment_method.get("bark strip")
        index_lopping_pruning = self.mapping_treatment_method.get("lopping / pruning")

        # set out conditions:
        # Condition 1: adult, landscape
        filtered_rows = norms[(norms[:, 3] == index_adult) & (norms[:, 5] == index_landscape)]
        if filtered_rows.size == 0: # empty
            pass
        elif filtered_rows.shape[0] == 1:
            return filtered_rows
        # Condition 1a: ring bark
        elif filtered_rows[(norms[:, 4] == index_ring_bark)].shape[0] == 1: # return the only row
            return filtered_rows[(norms[:, 4] == index_ring_bark)][0]
        elif filtered_rows[(norms[:, 4] == index_ring_bark)].shape[0] > 1: # return the first row
            return filtered_rows[(norms[:, 4] == index_ring_bark)][0]
        # Condition 1b: bark strip
        elif filtered_rows[(norms[:, 4] == index_bark_strip)].shape[0] == 1:
            return filtered_rows[(norms[:, 4] == index_bark_strip)][0]
        elif filtered_rows[(norms[:, 4] == index_bark_strip)].shape[0] > 1:
            return filtered_rows[(norms[:, 4] == index_bark_strip)][0]
        # Condition 1c: lopping/pruning
        elif filtered_rows[(norms[:, 4] == index_lopping_pruning)].shape[0] == 1:
            return filtered_rows[(norms[:, 4] == index_lopping_pruning)][0]
        elif filtered_rows[(norms[:, 4] == index_lopping_pruning)].shape[0] > 1:
            return filtered_rows[(norms[:, 4] == index_lopping_pruning)][0]

        # Condition 2: adult, riparian
        filtered_rows = norms[(norms[:, 3] == index_adult) & (norms[:, 5] == index_riparian)]
        if filtered_rows.size == 0:
            pass
        elif filtered_rows.shape[0] == 1:
            return filtered_rows
        # Condition 2a: felling
        elif filtered_rows[(norms[:, 4] == index_felling)].shape[0] == 1:
            return filtered_rows[(norms[:, 4] == index_felling)][0]
        elif filtered_rows[(norms[:, 4] == index_felling)].shape[0] > 1:
            return filtered_rows[(norms[:, 4] == index_felling)][0]
        # Condition 2b: bark strip
        elif filtered_rows[(norms[:, 4] == index_bark_strip)].shape[0] == 1:
            return filtered_rows[(norms[:, 4] == index_bark_strip)][0]
        elif filtered_rows[(norms[:, 4] == index_bark_strip)].shape[0] > 1:
            return filtered_rows[(norms[:, 4] == index_bark_strip)][0]
        # Condition 2c: lopping/pruning
        elif filtered_rows[(norms[:, 4] == index_lopping_pruning)].shape[0] == 1:
            return filtered_rows[(norms[:, 4] == index_lopping_pruning)][0]
        elif filtered_rows[(norms[:, 4] == index_lopping_pruning)].shape[0] > 1:
            return filtered_rows[(norms[:, 4] == index_lopping_pruning)][0]
        # Condition 3: lopping/pruning
        if norms[(norms[:, 4] == index_lopping_pruning)].shape[0] == 1:
            return norms[(norms[:, 4] == index_lopping_pruning)][0]
        elif norms[(norms[:, 4] == index_lopping_pruning)].shape[0] > 1:
            return norms[(norms[:, 4] == index_lopping_pruning)][0]

        # Condition 3a: other (1st choice)
        else:
            return norms[0]

    # filter and get the ppd from clearning norm data
    def get_person_day_factor(self,norm_filter_data_row):
        ## norm filter_data has the folowing columns:
        # # 0. density (densified values)
        # # 1. initial
        # # 2. growth_form
        # # 3. age
        # # 4. riparian

        # get the absolute difference of the density values,
        density_differance = np.abs(norm_filter_data_row[0] - self.support_clearing_norm_data[:, 0])
        # find the minimum of this difference and subtract again: this is to reduce all values to near 0, as the initial difference can contain values larger than 0 for density differences
        density_differance = np.abs(density_differance - np.min(density_differance))
        # now can use condition with < 1e-6 to get density rows
        density_differance = self.support_clearing_norm_data[density_differance < 1e-6]
      
        # Filter rows where all values exist, use same variable name, for optimization
        density_differance = density_differance[np.where(norm_filter_data_row[1] == density_differance[:, 1])[0]] # initial
        density_differance = density_differance[np.where(norm_filter_data_row[2] == density_differance[:, 2])[0]] # growth_form
        density_differance = density_differance[np.where(norm_filter_data_row[3] == density_differance[:, 3])[0]] # age
        density_differance = density_differance[np.where(norm_filter_data_row[4] == density_differance[:, 5])[0]] # riparian

        # selection process for which treatment frequency to use:
        if density_differance.shape[0] == 1:
            # get final person days
            person_days_per_hectare = density_differance[0,-1] # last value is person day per hectare
        else:
            treatment_selection = self.treatment_selection(density_differance)
            person_days_per_hectare = treatment_selection[-1]
        return person_days_per_hectare

    # calculate person days
    def calculate_person_days(self,density_matrix, species_matrix, working_hours_per_day):
        """
        :param density:
        :param is_initial:
        :param growth_form:
        :param age:
        :param riparian:
        :param walk_time:
        :param drive_time:
        :param area:
        :param slope:
        :param working_hours_per_day:
        :return:
        """


        # concatenate and pad three extra columns with person day calculation
        person_days = np.concatenate((density_matrix, species_matrix,np.full((len(species_matrix[:,0]),3),0)), axis=1).astype(float)


        # # filter clearing norm data in this order:
        # # 1. density
        # # 2. growth_form
        # # 3. age
        # # 4. is_initial
        # # 5. riparian

        # # todo check this: check if cleared: np.isnan(row[:, [1, 3, 5, 6]]).any(), this is now done in the costing step
        mask = ~np.isnan(person_days[:,-3])  # Create a mask to filter rows with non-NaN values in the second column
        selected_columns = person_days[:, [0,2,3,4,5]]  # Select the desired columns
        person_days[mask,-3] = np.apply_along_axis(self.get_person_day_factor, axis=1, arr=selected_columns[mask])#person_days[mask,[0,2,3,4,5]])
        # obtain person days:
        person_days[:,[-2,-1]] = PersonDays().calculate_person_days(person_days[:,[-3,6,7,8,9]],working_hours_per_day.values)

        return person_days[:,[-2,-1]]

    # cost formula:
    def cost_formula(self,cost_matrix):
        # data in cost_matrix:
        # 0. person days normal
        # 1. person days adjusted
        # 2. team size
        # 3. team cost
        # 4. user defined daily cost
        # 5. vehicle cost
        ## formula: note if team size or cost per day is 0, then set to 1 for this formula to work. # this has been done in the preprocessing
        cost = cost_matrix[:,1] * cost_matrix[:,3] / cost_matrix[:,2] + cost_matrix[:,5] * cost_matrix[:,0] + cost_matrix[:,4] * cost_matrix[:,0]

        return cost

    # fuel cost per compartment
    def fuel_formula(self,fuel_matrix):
        # data in fuel_matrix:
        # 0. fuel cost per hour
        # 1. drive time
        # Fuel_cost_per_compartment = Fuel_cost_hour x Drive_time / 10
        return fuel_matrix.iloc[:,0]*fuel_matrix.iloc[:,1]/10 # pandas friendly

    # sort compartment individual entries costing entries wrt prioritization
    def sort_costing_prioritizations(self,cost_matrix,budget):
        # this function will convert the costing to a dataframe sort it and then determine if the entries are costed or not
        # Convert the numpy array to a DataFrame, with the following order of column headers
        cost_df = pd.DataFrame(cost_matrix, columns=['priority', 'density', 'cost'])

        # Add a column called 'is_costed' with initial value 0
        cost_df['is_costed'] = 0

        # Sort the DataFrame based on priority (descending), density (ascending), and cost (ascending)
        cost_df.sort_values(by=['priority', 'density', 'cost'], ascending=[False, True, True], inplace=True)


        # Perform the budget calculation and update 'is_costed' column
        cost_values = cost_df["cost"].values
        temp_cost = np.column_stack((cost_values, np.zeros_like(cost_values)))
        for index, value in enumerate(temp_cost[:,0]):
            if budget >= value:
                temp_cost[index,1] = 1
                budget -= value
            else:
                continue

        cost_df["is_costed"] = temp_cost[:,1]

        # Check if there are remaining entries and revert the sorting
        cost_df.sort_index(inplace=True)

        # Output the final DataFrame
        return cost_df.values,budget

    # sort compartment individual entries costing entries wrt prioritization
    def sort_fuel_compartment_costing(self,fuel_cost_all_compartments):
        # This function takes a dataframe with compt_id and fuel_cost as columns
        # the fuel cost is determined by dividing by the number of duplicate entries for that compartment according to miu entries

        # Step 1: Group the dataframe by 'compt_id' and take the first value of 'fuel_cost'
        grouped = fuel_cost_all_compartments.groupby('compt_id')['fuel_cost'].agg('first').reset_index()

        # Step 2: Merge the aggregated values back to the original dataframe
        df_merged = pd.merge(fuel_cost_all_compartments, grouped, on='compt_id')

        # Step 3: Calculate the count of duplicates for each unique 'compt_id' value
        duplicates_count = fuel_cost_all_compartments.groupby('compt_id').size().reset_index(name='count')

        # Step 4: Merge the count of duplicates back to the merged dataframe
        df_merged = pd.merge(df_merged, duplicates_count, on='compt_id')

        # Step 5: Calculate the divided cost
        df_merged['divided_cost'] = df_merged['fuel_cost_x'] / df_merged['count']

        # Step 6: Assign the divided cost back to the 'fuel_cost' column in the dataframe
        df_merged['fuel_cost_y'] = df_merged['divided_cost']

        # Step 7: Remove the extra columns
        df_merged = df_merged[['compt_id', 'fuel_cost_y']].rename(columns={'fuel_cost_y': 'fuel_cost'})

        return df_merged

    # get costing
    def calculate_costing(self,person_days_matrix, species_matrix,treatment_frequency_mask, current_budget,fuel_cost_per_compartment):
        # prepare data -  pad with 8 extra columns for indexes 18 onwards
        costing_matrix = np.concatenate((person_days_matrix, species_matrix,treatment_frequency_mask,np.full((len(species_matrix[:,0]),8),0)), axis=1).astype(float)

        # populate team size if initial or followup
        costing_matrix[:, -8] = np.where(costing_matrix[:, 4], costing_matrix[:, 7], costing_matrix[:, 9])
        # populate team cost per day if initial or followup
        costing_matrix[:, -7] = np.where(costing_matrix[:, 4], costing_matrix[:, 8], costing_matrix[:, 10])

        # costing algorithm
        costing_matrix[:,-3] = self.cost_formula(costing_matrix[:,[1,2,-8,-7,13,11]])

        # test apply masking
        ## cleared
        costing_matrix[:, -3] = np.where(costing_matrix[:, 3], np.nan, costing_matrix[:, -3])
        ## treatment frequency
        costing_matrix[:, -3] = np.where(costing_matrix[:, 17], np.nan, costing_matrix[:, -3])
        # print(costing_matrix[:,-3])

        # fuel cost per compartment (calculated as a preprocessing step):
        costing_matrix[:,-6] = fuel_cost_per_compartment["fuel_cost"].values
        # add the fuel cost onto the current costing: optimal, budgets 1,2,3,and 4 will be determine from the optimal costing in the selection process
        costing_matrix[:,-3] = costing_matrix[:,-3] + costing_matrix[:,-6]

        # prioritization cost sorting
        ## make a dataframe which has the following columns and also returned: priority, density, cost   (, is_costed: created in the function)
        temp_costing_matrix,current_budget = self.sort_costing_prioritizations(costing_matrix[:,[15,0,-3]],current_budget)

        # TODO: TEST update and re-assign the following conditions:
        # 0. cost (if is_costed, then keep cost, otherwise np.nan)
        costing_matrix[:,-3] = np.where(temp_costing_matrix[:,3],costing_matrix[:,-3],np.nan)
        # 1. density (if is_costed, then use density reduction, otherwise densify)
        costing_matrix[:,-5] = np.where(temp_costing_matrix[:,3],costing_matrix[:,6],costing_matrix[:,5])
        # 2. person days adjusted (if is_costed, then use person days adjusted, otherwise 0)
        costing_matrix[:,-4] = np.where(temp_costing_matrix[:,3],costing_matrix[:,2],0)
        # 3. is cleared (check if density < maintenance level, then is cleared, otherwise not cleared)
        costing_matrix[:,-2] = np.where(costing_matrix[:,-5] < costing_matrix[:,14],1,0)
        # 4. is initial
        costing_matrix[:,-1] = np.where(temp_costing_matrix[:,3],0,costing_matrix[:,4])

        return costing_matrix[:,[-5,-4,-3,-2,-1]],current_budget,temp_costing_matrix[:,3]

    ## 3. Main costing loop
    def calculate_budgets(self):
        # Set out initial data parameters:
        ## number of budget years
        number_of_years = self.mucp_input_file.planning_years.item()
        working_day_hours = self.mucp_input_file.operations_working_day_hours.item()
        working_year_days = self.mucp_input_file.operations_working_year_days.item()
        if number_of_years > 50:
            return
        if working_day_hours > 16:
            return
        if working_year_days > 300:
            return

        ## preprocess miu data to include a treatment/Initial boolean column
        miu_linked_species_data = self.miu_linked_species_file.data.copy()
        nbal_linked_species_data = self.nbal_linked_species_file.data.copy()
        gis_mapping = self.gis_file.data.copy()

        # create starting species dataframe:
        nbal_initial_setup = pd.merge(gis_mapping[gis_mapping['nbal_id'].notnull()], nbal_linked_species_data, on='nbal_id', how='left')
        miu_initial_setup = pd.merge(gis_mapping[gis_mapping['nbal_id'].isnull()], miu_linked_species_data, on='miu_id', how='left')
        initial_setup = pd.concat([nbal_initial_setup,miu_initial_setup]).reset_index(drop=True)
        initial_setup = initial_setup.rename(columns={'idenscode':'density'})

        ## add an Initial data frame with constants
        ## Apply lambda function to MIU_dataframe to set up constants:
        # initial treatment (nbal data)
        initial_setup['initial'] = initial_setup['nbal_id'].apply(lambda row: self.is_initial_treatment(row))
        # riparian (miu data)
        initial_setup['riparian'] = initial_setup['miu_id'].apply(lambda row: self.get_riparian(row))
        # compartment data
        initial_setup[['slope', 'walk_time', 'drive_time', 'costing_code','growth_condition']] = initial_setup['compt_id'].apply(lambda row: self.get_compartment_info(row))
        # species information
        initial_setup[['initial_density_reduction','followup_density_reduction','densification','treatment_frequency','growth_form']] = initial_setup['species'].apply(lambda row: self.get_species_constants(row))
        # flow reduction factor
        initial_setup['flow_reduction_factor'] = initial_setup.apply(lambda row: self.get_flow_reduction_factor(row['compt_id'], row['age'], row['species'],row['growth_condition']), axis=1)
        # costing model
        initial_setup[['cost_plan','initial_team_size','initial_cost_per_day','followup_team_size','followup_cost_per_day','vehicle_cost_per_day','fuel_cost_per_hour','daily_cost','maintenance_level']] = initial_setup['costing_code'].apply(lambda row: self.get_costing_model(row))

        # process rows to extract error rows and remove from todo: return errors to viewer
        error_rows_df = initial_setup[initial_setup[['compt_id', 'area', 'costing_code', 'cost_plan', 'density','initial','growth_form','walk_time','drive_time','area']].isna().any(axis=1)].copy()
        initial_setup.dropna(subset=['compt_id', 'area', 'costing_code', 'cost_plan', 'density','initial','growth_form','walk_time','drive_time','area'], inplace=True)
        # multiply the density reduction factors by -1:
        initial_setup['initial_density_reduction'] = initial_setup['initial_density_reduction'] * -1
        initial_setup['followup_density_reduction'] = initial_setup['followup_density_reduction'] * -1


        # get slope factor
        initial_setup['slope_factor'] = initial_setup['slope'].apply(lambda row: self.get_slope_factor(row))
        # get compartment prioritization:
        prioritization = self.get_prioritization()
        initial_setup['prioritization'] = initial_setup["compt_id"].apply(lambda row: self.set_prioritization(row,prioritization))
        # set the mean annual runoff for the flow calculation per compartment
        # first check if the mean annual runoff is in the prioritization file:
        possible_MAR_combinations = ["mean_annual_runoff", "mean annual runoff", "mar", "annual runoff", "mean runoff", "mean_runoff", "annual_runoff", "runoff"]
        matching_columns = [col for col in self.compartment_priority_file.data.columns if col in possible_MAR_combinations]
        initial_setup['mean_annual_runoff'] = initial_setup["compt_id"].apply(lambda row: self.set_mean_annual_runoff_per_compartment(row, self.compartment_priority_file.data,matching_columns))

        # for costing to work, set the team size and team cost per day to 1 if it was 0 (initial and followup):
        initial_setup['initial_team_size'] = initial_setup['initial_team_size'].where(initial_setup['initial_team_size'] != 0, 1)
        initial_setup['initial_cost_per_day'] = initial_setup['initial_cost_per_day'].where(initial_setup['initial_cost_per_day'] != 0, 1)
        initial_setup['followup_team_size'] = initial_setup['followup_team_size'].where(initial_setup['followup_team_size'] != 0, 1)
        initial_setup['followup_cost_per_day'] = initial_setup['followup_cost_per_day'].where(initial_setup['followup_cost_per_day'] != 0, 1)

        # # condition the data to use indexes and not strings for optimization: i.e. insert unique numbers corresponding to string value in the column
        initial_setup.iloc[:, 18] = initial_setup.iloc[:, 18].map(self.mapping_growth_form)
        initial_setup.iloc[:, 6] = initial_setup.iloc[:, 6].map(self.mapping_age)
        initial_setup.iloc[:, 8] = initial_setup.iloc[:, 8].map(self.mapping_riparian)
        time_step_0 = initial_setup[['density']].copy()
        time_step_0['person_days_normal'] = 0
        time_step_0['person_days'] = 0
        time_step_0['flow'] = 0
        time_step_0['cost'] = 0
        time_step_0['is_cleared'] = False
        time_step_0['initial'] = initial_setup[['initial']].copy()

        # get the data for the actual budgets: for each budget over the years of simulation
        financial_funds = self.get_yearly_funds(number_of_years)
        financial_funds_values = financial_funds.values # make the costing loop faster

        # unique treatment frequencies:
        treatment_frequencies = initial_setup['treatment_frequency'].unique().tolist()
        if 12 not in treatment_frequencies:
            treatment_frequencies.append(12)

        ## total time steps
        time_steps = number_of_years * 12 + 1 # for final calculation

        # Create a range of row indices from 0 to (y-1)
        row_indices = np.arange(time_steps)
        # Create a dictionary of column data using broadcasting
        data = {value: row_indices % value for value in treatment_frequencies}
        # Create the dataframe from the dictionary
        time_steps_costing = pd.DataFrame(data)
        # replace all non zero values with nan
        time_steps_costing = time_steps_costing.mask(time_steps_costing.ne(0), np.nan)
        # Replace 0 values with column names converted to integers
        time_steps_costing = time_steps_costing.replace(0, {col: int(col) for col in time_steps_costing.columns})
        # Insert NaN into all values of the 0 index row
        time_steps_costing.iloc[0, :] = np.nan
        time_steps_costing = time_steps_costing.values

        ## calculate fuel cost per compartment: (fixed costs)
        fuel_cost_all_compartments = self.fuel_formula(initial_setup[['fuel_cost_per_hour','drive_time']]).rename("fuel_cost")
        fuel_cost_all_compartments = pd.concat([initial_setup["compt_id"], fuel_cost_all_compartments], axis=1)

        fuel_cost_per_compartment = self.sort_fuel_compartment_costing(fuel_cost_all_compartments)

        # numpy alternative:
        # Step 1: Create a numpy array filled with NaN with dimensions of (all budgets and +1 for optimal,number of timesteps, number of rows, number of columns)
        row_len = initial_setup.shape[0]  # Length of each column
        final_budgets = np.full((len(financial_funds.columns),time_steps+1, row_len, 7), np.nan)

        # Step 2: Insert data from timestep0 into the 0 index for timesteps
        final_budgets[:,0,:,:] = time_step_0.values
        # Step 3a: loop over the budgets:
        for budget_index,budget_name in enumerate(financial_funds.columns):
            # financial_fund = financial_funds[budget_name]
            financial_fund = financial_funds_values[:,budget_index]
            # Step 3b: Iterate over each timestep
            for index,timestep in enumerate(time_steps_costing):
                if index%12 == 0: # set current budgets for each new year
                    current_budget = financial_fund[int(index/12)]
                    print(budget_name,index/12,current_budget)
                if np.isnan(timestep).all(): # if no timestep, then copy previous timestep data
                    final_budgets[budget_index,index+1] = final_budgets[budget_index,index]
                    continue
                else: # otherwise calculate new timestep data
                    # put condition for is all timesteps is 0
                    # Step 4: Create a temporary numpy array
                    temp_array = np.full((row_len, 9), np.nan)

                    # assign previous timestep data:
                    previous_timestep = final_budgets[budget_index,index]
                    temp_array[:,0:7] = previous_timestep

                    # current timestep
                    index+=1

                    # obtain if the treatment frequency is in this timestep and should be calculated
                    individual_treatment_frequencies = timestep[~np.isnan(timestep)]
                    treatment_frequencies_mask = initial_setup["treatment_frequency"].values
                    treatment_frequencies_mask = np.isin(treatment_frequencies_mask, individual_treatment_frequencies).T.reshape(-1, 1)

                    temp_array[:,[-2,-1]] = self.calculate_density(temp_array[:,[0,5,6]],initial_setup[["densification",'initial_density_reduction','followup_density_reduction']].values)


                    # step 6: calculate person days
                    temp_array[:,[1,2]] = self.calculate_person_days(temp_array[:,[0,5,6]],initial_setup[["growth_form", "age", "riparian", "walk_time", "drive_time", "area", "slope_factor"]].values,self.mucp_input_file.operations_working_day_hours)


                    # step 7: calculate costing
                    temp_array[:,[0,2,4,5,6]],current_budgets,is_costed = self.calculate_costing(
                        temp_array[:,[0,1,2,-4,-3,-2,-1]],
                        initial_setup[["initial_team_size","initial_cost_per_day","followup_team_size","followup_cost_per_day","vehicle_cost_per_day","fuel_cost_per_hour","daily_cost","maintenance_level","prioritization","drive_time"]].values,
                        treatment_frequencies_mask,
                        current_budget,
                        fuel_cost_per_compartment,
                    )

                    # step 8: calculate flow
                    temp_array[:,3] = self.calculate_flow(temp_array[:,[0,6]], initial_setup[["area","riparian","flow_reduction_factor","mean_annual_runoff","densification",'initial_density_reduction','followup_density_reduction']].values,is_costed)

                    # Assign temp_array to the corresponding timestep in data_array
                    final_budgets[budget_index,index] = temp_array[:, 0:7]
        # condense monthly results into yearly results:
        final_budgets_yearly = [
            [
                pd.DataFrame( np.concatenate((initial_setup[["compt_id","miu_id","nbal_id","area",]].values, j[:, [0, 2, 3, 4]]), axis=1),columns=[
                    "output_compartment_id",
                    "output_miu_id",
                    "output_nbal_id",
                    "output_area",
                    "output_density",
                    "output_person_days",
                    "output_flow",
                    "output_cost"]).assign(output_budget_option="Optimal" if i == 4 else "Budget_"+str(i+1),output_year = index+1) for index,j in enumerate(final_budgets[i, ::12, :, :])
            ]
            for i in range(len(final_budgets[:,0,0,0]))
        ]
        # final_budgets_yearly = [pd.DataFrame(columns=["density","person_days normal","person_days adjusted","flow","cost","is_cleared","initial"]) for data in range(final_budgets[:,0,0,0])]
        return final_budgets_yearly