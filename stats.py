import pandas as pd

# ==========================
# Central Tendency
# ==========================
def central_tendency(df, numeric_cols):
    """
    Returns mean, median, and mode for numeric columns
    """
    if not numeric_cols:
        return pd.DataFrame()
    
    mean = df[numeric_cols].mean().rename("Mean")
    median = df[numeric_cols].median().rename("Median")
    mode = df[numeric_cols].mode().iloc[0] if not df[numeric_cols].mode().empty else pd.Series()
    
    summary = pd.concat([mean, median, mode], axis=1)
    return summary

# ==========================
# Measures of Spread
# ==========================
def measures_of_spread(df, numeric_cols):
    """
    Returns standard deviation, variance, min, max, and quartiles
    """
    if not numeric_cols:
        return pd.DataFrame()
    
    std = df[numeric_cols].std().rename("Std")
    var = df[numeric_cols].var().rename("Variance")
    min_val = df[numeric_cols].min().rename("Min")
    max_val = df[numeric_cols].max().rename("Max")
    q1 = df[numeric_cols].quantile(0.25).rename("Q1")
    q3 = df[numeric_cols].quantile(0.75).rename("Q3")
    
    spread = pd.concat([std, var, min_val, max_val, q1, q3], axis=1)
    return spread

# ==========================
# Categorical Counts
# ==========================
def categorical_counts(df, cat_cols):
    """
    Returns counts for categorical columns
    """
    counts = {}
    for col in cat_cols:
        counts[col] = df[col].value_counts()
    return counts

