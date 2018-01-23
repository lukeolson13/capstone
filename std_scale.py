#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

__author__ = "Luke Olson"

class StdScale(BaseEstimator, TransformerMixin):
    """
    A generic class
    """

    def __init__(self, std=True, scale=True):
        """
        Constructor
        """
        self.std = std
        self.scale = scale

    def _std_f(self):
        std_mask = (self.df_new.dtypes == int) | (self.df_new.dtypes == np.float64) # only standardize numbers that are not associated with time features
        std_cols = self.df_new.columns[std_mask]
        ss = StandardScaler()
        self.df_new[std_cols] = ss.fit_transform(self.df_new[std_cols])

    def _scale_f(self):
        sc_mask = (self.df_new.dtypes == np.float32) # only scale time features
        sc_cols = self.df_new.columns[ sc_mask ]
        min_time = self.df_new[sc_cols].min().values.min()
        max_time = self.df_new[sc_cols].max().values.max()
        for col in sc_cols:
            # scale all time features using the same two values, so equivalent values reference the same date across columns
            self.df_new[col] = (self.df_new[col] - min_time) / (max_time - min_time)

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        self.df = df
        if not self.std and not self.scale:
            return self.df
        self.df_new = self.df.copy()
        if self.std:
            self._std_f()
        if self.scale:
            self._scale_f()
        return self.df_new

if __name__ == "__main__":
    pass