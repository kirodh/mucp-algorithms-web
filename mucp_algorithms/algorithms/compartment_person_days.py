# import numpy as np

def calculate_normal_person_days(PPD: float, area: float) -> float:
    """
    Calculate normal and adjusted person-days based on input data.

    Formula:
        PD_normal = PPD × Area
        PD_adjusted = PD_normal × working_hours / (working_hours - (2 × (walk_time + drive_time) / 60)) × slope_factor

    Parameters
    ----------
    person_day_data : np.ndarray
        Array with shape (n, 5), where columns are:
            0 = person_day_factor (PPD)
            1 = walk_time (minutes)
            2 = drive_time (minutes)
            3 = area
            4 = slope_factor
    working_hours_per_day : int, optional
        Default = 8 (standard working hours per day).

    Returns
    -------
    np.ndarray
        Array with shape (n, 2), where:
            [:,0] = normal person-days
            [:,1] = adjusted person-days
    """

    # Step 1: normal person-days
    normal_person_days = PPD * area

    return normal_person_days

def calculate_adjusted_person_days(PPD_normal: float, walk_time: float, drive_time: float, slope_factor: float, working_hours_per_day: float) -> float:
    """
    Calculate normal and adjusted person-days based on input data.

    Formula:
        PD_normal = PPD × Area
        PD_adjusted = PD_normal × working_hours / (working_hours - (2 × (walk_time + drive_time) / 60)) × slope_factor

    Parameters
    ----------
    person_day_data : np.ndarray
        Array with shape (n, 5), where columns are:
            0 = person_day_factor (PPD)
            1 = walk_time (minutes)
            2 = drive_time (minutes)
            3 = area
            4 = slope_factor
    working_hours_per_day : int, optional
        Default = 8 (standard working hours per day).

    Returns
    -------
    np.ndarray
        Array with shape (n, 2), where:
            [:,0] = normal person-days
            [:,1] = adjusted person-days
    """


    # Step 2: adjusted person-days
    denominator = working_hours_per_day - (2 * (walk_time + drive_time) / 60)

    adjusted_person_days = slope_factor * (PPD_normal * working_hours_per_day) / denominator

    return adjusted_person_days
