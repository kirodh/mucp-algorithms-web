"""
Purpose: Flow algorithm of the MUCP tool. Check out the algorithm file for more information.
    This code is optimized to use vectorization for speed. Flow only works when runoff is provided.
Author: Kirodh Boodhraj
"""

# main flow algorithm
def calculate_flow(
    MAR: float,
    area: float,
    density: float,
    density_factor: float,
    flow_reduction_factor: float,
    riparian: bool
) -> float:
    """
    Calculate compartment flow based on Mean Annual Run off (MAR), area, density, and modifiers.

    Is calculated only if Mean Annual Runoff is available.

    Formula:
        1. MAR_density = MAR (mm/year) × area × density
        2. Flow_reduction = MAR_density × (100 - density_factor) / 100
        3. If riparian = True → Flow_reduction *= 1.5
        4. Flow = Flow_reduction × flow_reduction_factor

    Parameters
    ----------
    MAR : float
        Mean Annual Runoff (mm/year).
    area : float
        Compartment area (e.g., hectares).
    density : float
        Species density (0–100%).
    density_factor : float
        Factor for reduction due to density (percentage).
    flow_reduction_factor : float
        Global flow reduction factor.
    riparian : bool
        Whether the compartment is riparian (multiplies flow by 1.5).

    Returns
    -------
        Computed flow value.
    """

    # Step 1 & 2: base flow
    base_flow = MAR * area * density * (100 - density_factor) / 100
    flow = base_flow * flow_reduction_factor

    # Step 3: riparian adjustment
    if riparian == "riparian":
        flow *= 1.5

    # return float(flow)
    return flow
