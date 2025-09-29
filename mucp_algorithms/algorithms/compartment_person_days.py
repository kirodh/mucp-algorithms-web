"""
Purpose: Person day algorithm of the MUCP tool. Check out the algorithm file for more information.
    This code is optimized to use vectorization for speed.
Author: Kirodh Boodhraj
"""

# Main Normal Person Day algorithm
def calculate_normal_person_days(PPD: float, area: float) -> float:
    """
    Calculate normal and adjusted person-days based on input data.

    Formula:
        PD_normal = PPD × Area
        PD_adjusted = PD_normal × working_hours / (working_hours - (2 × (walk_time + drive_time) / 60)) × slope_factor

    """

    # Step 1: normal person-days
    normal_person_days = PPD * area

    return normal_person_days


# Main Person Day algorithm (aka adjusted person days)
def calculate_adjusted_person_days(PPD_normal: float, walk_time: float, drive_time: float, slope_factor: float, working_hours_per_day: float) -> float:
    """
    Calculate normal and adjusted person-days based on input data.

    Formula:
        PD_normal = PPD × Area
        PD_adjusted = PD_normal × working_hours / (working_hours - (2 × (walk_time + drive_time) / 60)) × slope_factor

    """


    # Step 2: adjusted person-days
    denominator = working_hours_per_day - (2 * (walk_time + drive_time) / 60)

    adjusted_person_days = slope_factor * (PPD_normal * working_hours_per_day) / denominator

    return adjusted_person_days
