#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from shrink_functions import X_y

__author__ = "Luke Olson"

class Split(BaseEstimator, TransformerMixin):
    """
    Break data into features and targets, and train/test split by time or randomly.
    """

    def __init__(self, non_feature_cols, target_col, split_by_time, date_col=None, split_date=None):
        """
        Initializer
        Inputs:
            non_feature_cols - columns that are either targets or result in data leakage
            target_col - target column
            split_by_time - whether or not to split by time. If false, a random split will occur
            date_col - column with dates to split on. Only required if split_by_time==True
            split_date - date to split data on. Only required if split_by_time==True
        """
        self.non_feature_cols = non_feature_cols
        self.target_col = target_col
        self.split_by_time = split_by_time
        self.date_col = date_col
        self.split_date = split_date

    def time_split(self, df):
        """
        Splits dataframe by date into train/test sets
        Inputs:
            df - dataframe
        Returns:
            X_train - training features
            X_test - testing features
            y_train - training targets
            y_test - testing targets
        """
        df_train = df[ df[self.date_col] < self.split_date ]
        df_test = df[ df[self.date_col] >= self.split_date ]
        X_train, y_train = X_y(df_train, self.non_feature_cols, self.target_col)
        X_test, y_test = X_y(df_test, self.non_feature_cols, self.target_col)

        return X_train, X_test, y_train, y_test

    def fit(self, df_orig, cust_table_orig, y=None):
        """
        Placeholder fit method required by sklearn
        """
        return self

    def transform(self, df_orig, cust_table_orig):
        """
        Splits data in datframe in train/test data
        Inputs:
            df_orig - dataframe
            cust_table_orig - customer table
        Returns:
            X - features
            y - targets
            X_train - training features
            X_test - testing features
            y_train - training targets
            y_test - testing targets
        """
        df = df_orig.copy()
        cust_table = cust_table_orig.copy()
        df = df.join(cust_table[['cluster']], on='address1', how='left')
        X, y = self.X_y(df)
        if self.split_by_time:
            X_train, X_test, y_train, y_test = self.time_split(df)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=X.cluster.values)
        return X, y, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    pass