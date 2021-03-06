#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

__author__ = "Luke Olson"

class CreateLag(BaseEstimator, TransformerMixin):
    """
    Create lag columns to capture target variables at previous time points.
    """

    def __init__(self, lag_periods, col_filters, date_col, lag_vars, col_name_suf, remove_nan_rows=True):
        """
        Initializer
        Inputs:
            lag_periods - how many time periods back to create lag columns for
            col_filters - heirarchal list of columns to filter down on when creating lag values
            date_col - date column to filter on to get previous time stamps
            lag_vars - columns of values to create lag variables with
            col_name_suf - suffix to append to newly created columns (help to distinguish between last
            remove_nan_rows - whether or not to remove dataframe rows that contain nan value
        """
        self.lag_periods = lag_periods
        self.col_filters = col_filters
        self.date_col = date_col
        self.lag_vars = lag_vars
        self.col_name_suf = col_name_suf
        self.remove_nan_rows = remove_nan_rows

    def _init_nans(self):
        """
        Initiate column(s) of nan values for your lag variables in the given dataframe
        """
        for period in range(1, self.lag_periods + 1):
            for lag_var in self.lag_vars:
                self.df['{}_lag{}{}'.format(lag_var, period, self.col_name_suf)] = np.nan

    def _set_lag_vals(self, comb_mask):
        """
        Sets lag values according to the last column in the heirarchy or columns
        Input:
            comb_mask - mask for filtered dataframe to create lag variables from
        """
        foo = self.df[ comb_mask ].sort_values(self.date_col, ascending=False)
        length = len(foo[self.date_col].unique()) # determine number of visits (because multiple item categories can be updated in a single visit)
        for period in range(1, self.lag_periods + 1):
            # skip if there's not enough data to create lag variables
            if length < period + 1:
                continue
            i = 0
            # create duplicate df, but with all indices shifted by the current 'period' number
            foo_shifted = foo.shift(-period)
            foo_grouped = foo.groupby(self.date_col).mean()
            for index, row in foo.iterrows():
                date = foo_shifted[ foo_shifted.index == index ][self.date_col].values[0]
                for lag_var in self.lag_vars:
                    lag_val = foo_grouped[ foo_grouped.index == date ][lag_var].values[0]
                    # set value
                    self.df.set_value(index, '{}_lag{}{}'.format(lag_var, period, self.col_name_suf), lag_val)
                i += 1
                if i + period == length:
                    break # back to period loop

    def _lag_rec(self, col_filters, mask=True):
        """
        Recursively loop through various heirarchaly ordered columns, grouping by date
        INPUTS:
            col_filters - columns to heiracrchally filter down on, with the last column being the one ultimately used
            mask - DO NOT CHANGE. Required to be True to maintain dataframe mask between recursive iterations
        """
        # begin with mask of all trues
        true_mask = np.ones(len(self.df), dtype=bool)
        loop_mask = mask & true_mask
        col_filter = col_filters[0]
        for val in self.df[ loop_mask ][col_filter].unique():
            val_mask = self.df[col_filter] == val
            comb_mask = loop_mask & val_mask

            if len(col_filters) > 1:
                #recursively update the remaining items' positions
                self._lag_rec(col_filters=col_filters[1:], mask=comb_mask)
            else:
                self._set_lag_vals(comb_mask)

    def lag(self):
        """
        Wrapper function to execute init_nans and lag_rec.
        """
        self._init_nans()
        self._lag_rec(self.col_filters)

    def nans(self):
        """
        Removes nan rows
        """
        self.df.dropna(axis=0, inplace=True)

    def fit(self, df, y=None):
        """
        Placeholder fit method required by sklearn
        """
        return self

    def transform(self, df):
        """
        Adds new lag columns to input dataframe
        Input:
            df - dataframe
        Returns:
            dataframe with new lag columns appended
        """
        self.df = df.copy()
        self.lag()
        if self.remove_nan_rows:
            self.nans()
        return self.df

if __name__ == "__main__":
    pass