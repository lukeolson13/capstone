import numpy as np
from sklearn.preprocessing import StandardScaler

def X_y(df, non_feature_cols, target_col):
    non_feature_data = df[non_feature_cols]
    features = list(set(df) - set(non_feature_cols))
    features.sort()
    X = df[features]
    y = non_feature_data[target_col]
    return X, y

def time_split(df, date_col, date, non_feature_cols, target_col):
    df_train = df[ df[date_col] < date ]
    df_test = df[ df[date_col] >= date ]
    X_train, y_train = X_y(df_train, non_feature_cols, target_col)
    X_test, y_test = X_y(df_test, non_feature_cols, target_col)
    return X_train, X_test, y_train, y_test

def std_f(X_std):
    std_mask = (X_std.dtypes == int) | (X_std.dtypes == np.float64) # only standardize numbers that are not associated with time features
    std_cols = X_std.columns[std_mask]
    ss = StandardScaler()
    X_std[std_cols] = ss.fit_transform(X_std[std_cols])
    return X_std

def scale_f(X_sc):
    sc_mask = (X_sc.dtypes == np.float32) # only scale time features
    sc_cols = X_sc.columns[sc_mask]
    min_time = X_sc[sc_cols].min().values.min()
    max_time = X_sc[sc_cols].max().values.max()
    for col in sc_cols:
        # scale all time features using the same two values, so equivalent values reference the same date across columns
        X_sc[col] = (X_sc[col] - min_time) / (max_time - min_time)
    return X_sc

def ss(X_ss, std=True, scale=True):
    if not std and not scale:
        return
    X_new = X_ss.copy()
    if std:
        X_new = std_f(X_new)
    if scale:
        X_new = scale_f(X_new)
    return X_new