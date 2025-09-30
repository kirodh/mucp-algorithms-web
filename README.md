# MUCP Algorithms

The **Management Unit Control Plan (MUCP) Algorithms Web** project provides core algorithms used in the MUCP decision-support tool.

These algorithms are designed to assist with ecological, economic, and operational planning for invasive species management.  

---

## 📖 Algorithms Overview

The MUCP tool implements **five core algorithms**:

1. Density
2. Prioritization
3. Person Days (Two parts)
4. Costing (Main algorithm to call all others)
5. Flow (Only calculated if Mean Annual Runoff is selected in the compartment priority)

Please see the algorithm manual which contains a more in depth description of the algorithms. It can be found in the docs folder.


### 1. Density Algorithm

**Purpose:**  
Calculates species density after accounting for densification or reduction factors.  

**Formula:**  

`Density = D_initial × (100 + F_densification) / 100`


- `D_initial`: Initial species density  
- `F_densification`: Densification factor (negative values act as reduction factors)  

---

### 2. Prioritization Algorithm

**Purpose:**  
Determines the priority of compartments based on multiple weighted factors (e.g., elevation, aggression, slope).  

**Process:**  
1. Read the **compartment prioritization file** (each row = compartment, each column = prioritization variable).  
2. Match column headers to reference prioritization fields.  
3. Map each variable to **user-defined weightings** from the input file (weights usually between 0–1).  
4. Identify the **category** for each variable based on ranges in the support priority tables.  
5. Assign a **priority value** for the category.  
6. Multiply the category’s priority by the user’s weighting.  
7. Sum across all variables → this gives the **final prioritization score** for the compartment.  

**Example:**  
- Elevation priority = 1, weight = 0.9 → 0.9  
- Aggression priority = 2, weight = 0.2 → 0.4  
- Total prioritization = **1.3**  

---

### 3. Person Days Algorithm

**Purpose:**  
Estimates required person-days for operations, accounting for slope, travel time, and working hours.  

There are two calculations:

#### a. Normal Person Days


`PD_normal = PPD x Area`


- `PPD`: Productivity per person per day  
- `Area`: Compartment area  

#### b. Adjusted Person Days

`PD_adjusted = (PD_normal x H x S) / (H - 2 x (T_walk + T_drive)/(60))`

- `H`: Working hours per day  
- `T_walk`: Walking time (minutes)  
- `T_drive`: Driving time (minutes)  
- `S`: Slope factor  

---

### 4. Costing Algorithm

**Purpose:**  
Calculates project costs and propagates budgets over multiple years.  

**Inputs include:**  
- Person days (normal and adjusted)  
- Initial or follow-up **cost per day** and **team size**  
- Vehicle cost per day  
- Daily operating costs  
- Fuel costs distributed per species in each compartment  

**Cost Formula:**  

`Cost = (PD x C_day / T_size) + (C_vehicle x PD_normal) + (C_daily x PD_normal) + (C_fuel/miu)`

- `PD`: Adjusted person days  
- `C_day`: Initial/follow-up cost per day  
- `T_size`: Initial/follow-up team size  
- `C_vehicle`: Vehicle cost per day  
- `C_daily`: User-defined daily costs  
- `C_fuel/miu`: Distributed fuel cost per species  

**Outputs:**  
1. **Budgets** — projected per year, per plan (`plan_1` … `plan_4`) using simple interest.  
   Example:  

   ```json
   {
     2025: {"plan_1": 10000000.0, "plan_2": 7500000.0, "plan_3": 5000000.0, "plan_4": 2500000.0},
     2026: {"plan_1": 11000000.0, "plan_2": 8250000.0, "plan_3": 5500000.0, "plan_4": 2750000.0}
   }
   ```

2. **Costing results** — detailed yearly outputs with density, priority, person days, and costs linked to GIS mapping.

---

### 5. Flow Algorithm

**Purpose:**
Estimates flow reduction in compartments based on Mean Annual Runoff (MAR), density, and modifiers.

**Formula:**

1. Base MAR-density:


`MAR_density = MAR x Area x Density`


2. Flow reduction:

`Flow_reduction = MAR_density x (100 - F_density) / 100`

3. Riparian adjustment (if riparian = true):

`Flow_reduction = Flow_reduction x 1.5`

4. Final flow:

`Flow = Flow_reduction x F_flow`

* `MAR`: Mean annual runoff (mm/year)
* `Area`: Compartment area
* `Density`: Species density (%)
* `F_density`: Density reduction factor (%)
* `F_flow`: Global flow reduction factor

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/kirodh/mucp-algorithms-web.git
cd mucp-algorithms-web
```

### Standard install

```bash
pip install .
```

### Development install

If you want to modify the code while using it:

```bash
pip install -e .
```

This uses the `pyproject.toml` and `requirements.txt` to manage dependencies.

---

## 🧪 Examples

The `examples/` folder provides test case files and an example runner script.

* `example_run.py` reads input files from `examples/example_case_files/`.
* Users can adjust parameters in the Excel file:

  ```
  examples/example_case_files/MUCP_support_data_user_input.xlsx
  ```
* Running `example_run.py` will generate a sample budget and costing outputs.

---

## 🛠 Support & Debugging

For troubleshooting:

* Django Docs: [https://docs.djangoproject.com/](https://docs.djangoproject.com/)
* StackOverflow: [https://stackoverflow.com/](https://stackoverflow.com/)
* Youtube tutorials
* AI tools like ChatGPT for debugging assistance

---

## Code Authors
- Kirodh Boodhraj

## 🌟 Special Mentions

We would like to extend our gratitude to the following individuals who contributed their knowledge, support, and vision to the development of the MUCP tool:

- **Greg Forsyth** — for his invaluable input on the theory, communication, and clear description of how the tool works.  
- **Ryan Blanchard** — for his assistance in understanding the core concepts and the project as a whole.  
- **William Stafford** — for his dedicated support, deep understanding of the broader economy surrounding the tool, and for helping to shape future uses and features. His exceptional insights into diverse industries (forestry, industrial, waste, etc.) greatly enriched the tool.  
- **Andrew Wannenburgh** — for initially funding the tool, contributing a key algorithm, and providing ongoing support for its continued development.  
- **David Le Maitre** *(in memoriam)* — who brought a profound understanding of ecological processes and was instrumental in developing the core theory of the MUCP tool. This tool is partially dedicated to his legacy.  

---
## License
Open-source. 

---
## 🙏 Funding & Acknowledgements

This project was developed under the funding and support of:

South Africa Department of Forestry, Fisheries and the Environment (DFFE)
🌍 https://www.dffe.gov.za

Council for Scientific and Industrial Research (CSIR)
🌍 https://www.csir.co.za

If you make use of this code or incorporate it in research or applications, please reference and acknowledge DFFE and CSIR.

---

END