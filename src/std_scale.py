#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np
from shrink_functions import std_f, scale_f

__author__ = "Luke Olson"

class StdScale(BaseEstimator, TransformerMixin):
    """
    Standardize and scale data.
    """

    def __init__(self, std=True, scale=True):
        """
        Initializer
        Inputs:
            std - whether or not to standardize data
            scale - whether or not to scale data
        """
        self.std = std
        self.scale = scale

    def fit(self, df, y=None):
        """
        Placeholder fit method required by sklearn
        """
        return self

    def transform(self, df):
        """
        Standardizes and/or scales data
        Inputs:
            df - dataframe
        Returns:
            Updated dataframe
        """
        if not self.std and not self.scale:
            return df
        self.df_new = df.copy()
        if self.std:
            self.df_new = std_f(self.df_new)
        if self.scale:
            self.df_new = scale_f(self.df_new)
        return self.df_new

if __name__ == "__main__":
    pass