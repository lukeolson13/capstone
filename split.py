#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

__author__ = "Luke Olson"

class Split(BaseEstimator, TransformerMixin):
    """
    A generic class
    """

    def __init__(self, non_feature_cols, target_col, split_by_time, date_col=None, split_date=None):
        """
        Constructor
        """
        self.non_feature_cols = non_feature_cols
        self.target_col = target_col
        self.split_by_time = split_by_time
        self.date_col = date_col
        self.split_date = split_date

    def X_y(self, df):
        non_feature_data = df[self.non_feature_cols]
        features = list(set(df) - set(self.non_feature_cols))
        features.sort()
        X = df[features]
        y = non_feature_data[self.target_col]
        return X, y

    def time_split(self, df):
        df_train = df[ df[self.date_col] < self.split_date ]
        df_test = df[ df[self.date_col] >= self.split_date ]
        X_train, y_train = self.X_y(df_train)
        X_test, y_test = self.X_y(df_test)

        return X_train, X_test, y_train, y_test

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        X, y = self.X_y(df)
        if self.split_by_time:
            X_train, X_test, y_train, y_test = time_split(df)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=X.cluster.values)
        return X, y, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    pass