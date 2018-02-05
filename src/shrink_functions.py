import numpy as np
from sklearn.preprocessing import StandardScaler

def X_y(df, non_feature_cols, target_col):
    """
    Splits dataframe into features and targets
    Inputs:
        df - dateframe
        non_feature_cols - columns that are either targets or result in data leakage
        target_col - target column (y)
    Returns:
        X - features
        y - target
    """
    non_feature_data = df[non_feature_cols]
    features = list(set(df) - set(non_feature_cols))
    features.sort()
    X = df[features]
    y = non_feature_data[target_col]
    return X, y

def std_f(X_std):
    """
    Standardizes data (centers column about 0, with a standard deviation of 1)
    Inputs:
        X_std - feature column dataframe
    Returns:
        standardized feature column dataframe
    """
    std_mask = (X_std.dtypes == int) | (X_std.dtypes == np.float64) # only standardize numbers that are not associated with time features
    std_cols = X_std.columns[std_mask]
    ss = StandardScaler()
    X_std[std_cols] = ss.fit_transform(X_std[std_cols])
    return X_std

def scale_f(X_sc):
    """
    Scales data (between 0 and 1)
    Inputs:
        X_sc - feature column dataframe
    Returns:
        scaled feature column dataframe
    """
    sc_mask = (X_sc.dtypes == np.float32) # only scale time features
    sc_cols = X_sc.columns[sc_mask]
    min_time = X_sc[sc_cols].min().values.min()
    max_time = X_sc[sc_cols].max().values.max()
    for col in sc_cols:
        # scale all time features using the same two values, so equivalent values reference the same date across columns
        X_sc[col] = (X_sc[col] - min_time) / (max_time - min_time)
    return X_sc

def ss(X_ss, std=True, scale=True):
    """
    Wrapper to standardize and scale data
    Inputs:
        X_ss - feature column dataframe
        std - whether or not to standardize data
        scale - whether or not to scale data
    Returns:
        (potentially) updated feature column dataframe
    """
    if not std and not scale:
        return
    X_new = X_ss.copy()
    if std:
        X_new = std_f(X_new)
    if scale:
        X_new = scale_f(X_new)
    return X_new