import numpy as np

def calculate_species_density(initial_density, densification_factor):
    density = initial_density * (100 + densification_factor) / 100
    return np.clip(density, 0.0, 100.0)  # fast min/max
    # return float(np.clip(density, 0, 100))


# possibility fo using a df.clip(0,100) if all is df coming in here


