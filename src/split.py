#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from shrink_functions import X_y, time_split

__author__ = "Luke Olson"

class Split(BaseEstimator, TransformerMixin):
    """
    Break data into features and targets, and train/test split by time or randomly.
    """

    def __init__(self, cust_table, non_feature_cols, target_col, split_by_time, date_col=None, split_date=None):
        """
        Initializer
        Inputs:
            cust_table - customer table with clusters
            non_feature_cols - columns that are either targets or result in data leakage
            target_col - target column
            split_by_time - whether or not to split by time. If false, a random split will occur
            date_col - column with dates to split on. Only required if split_by_time==True
            split_date - date to split data on. Only required if split_by_time==True
        """
        self.cust_table = cust_table
        self.non_feature_cols = non_feature_cols
        self.target_col = target_col
        self.split_by_time = split_by_time
        self.date_col = date_col
        self.split_date = split_date

    def fit(self, df, y=None):
        """
        Placeholder fit method required by sklearn
        """
        return self

    def transform(self, df):
        """
        Splits data in datframe in train/test data
        Inputs:
            df - dataframe
        Returns:
            X - features
            y - targets
            X_train - training features
            X_test - testing features
            y_train - training targets
            y_test - testing targets
        """
        df_copy = df.copy()
        cust_table_copy = self.cust_table.copy()
        if self.split_by_time:
            #forecast
            cols_to_add = ['avg_UPC_per_visit', 'days_between_visits', 'cluster']
        else:
            cols_to_add = ['cluster']
        df_copy = df_copy.join(cust_table_copy[cols_to_add], on='address1', how='left')
        X, y = X_y(df_copy, self.non_feature_cols, self.target_col)
        if self.split_by_time:
            X_train, X_test, y_train, y_test = time_split(df_copy, self.date_col, self.split_date, self.non_feature_cols, self.target_col)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=X.cluster.values)
        return X, y, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    pass