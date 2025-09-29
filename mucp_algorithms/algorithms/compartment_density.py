"""
Purpose: Density algorithm of the MUCP tool. Check out the algorithm file for more information.
    This code is optimized to use vectorization for speed. Note that this code handles both density reduction and
    densification (- value must be provided).
Author: Kirodh Boodhraj
"""

import numpy as np

# main density algorithm
def calculate_species_density(initial_density, densification_factor):
    """

    Parameters
    ----------
    initial_density
    densification_factor

    Returns
    -------
    density value

    Formula for density is density_initial * (100 + densification_factor) / 100

    Note that densification_factor when negative is the same as the density reduction factor.
    """
    density = initial_density * (100 + densification_factor) / 100
    return np.clip(density, 0.0, 100.0)  # fast min/max
    # return float(np.clip(density, 0, 100))
    # possibility fo using a df.clip(0,100) if all is df coming in here


