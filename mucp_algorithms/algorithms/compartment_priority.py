import pandas as pd

def get_priorities(prioritization_data: pd.DataFrame, category_data: list ) -> pd.DataFrame:

    # Helper functions
    def get_priority_numeric(val, ranges):
        for low, high, priority in ranges:
            if low <= val <= high:
                return priority
        return 0  # default if not found

    def get_priority_text(val, allowed):
        val = str(val).lower().strip()
        for entry in allowed:
            if entry["value"] == val:
                return entry["priority"]
        return 0  # default if not found

    # Compute weighted score for each row
    def compute_score(row, categories):
        total = 0
        for cat in categories:
            col = cat["name"]
            if col not in row:  # skip if column missing
                continue
            if cat["type"] == "numeric":
                priority = get_priority_numeric(row[col], cat["ranges"])
            else:
                priority = get_priority_text(row[col], cat["allowed"])
            total += priority * cat["weight"]
        return total

    prioritization_data["prioritization"] = prioritization_data.apply(lambda r: compute_score(r, category_data), axis=1)

    try:
        return prioritization_data[["compt_id", "prioritization"]]
    except Exception as e:
        return None


