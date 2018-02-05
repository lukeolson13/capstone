#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import time

__author__ = "Luke Olson"

class DataClean(BaseEstimator, TransformerMixin):
    """
    Clean data - create new columns, drop others, and get rid of nan values.
    """

    def __init__(self, remove_nan_rows=True):
        """
        Initializer
        Input:
            remove_nan_rows - whether or not to remove all rows from dataframe that contain a nan value
        """
        self.remove_nan_rows = remove_nan_rows

    def _date_to_int(self, row, col):
        """
        Converts date to integer (seconds since epoch)
        Inputs:
            row - dataframe row (passed by apply function)
            col - data column to convert
        Returns:
            Integer version of date
        """
        index = self.df.columns.get_loc(col)
        date = row[index]
        return time.mktime(time.strptime(str(date), '%Y-%m-%d %H:%M:%S'))

    def data_type(self):
        """
        Change data types of certain columns, such as making dates datetimes
        """
        # datetime and date to int
        date_cols = ['visit_date', 'prev_visit_date', 'prev_item_move_date',
                     'last_edit_date', 'creation_date']
        for col in date_cols:
            # convert multiple time formats into single string format
            self.df[col] = pd.to_datetime(self.df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
            # make time features specific data type in order to distinguish from other numberic values
            self.df['{}_int'.format(col)] = self.df.apply(self._date_to_int, col=col, axis=1).astype(np.float32)
            # convert string format back into datetime
            self.df[col] = pd.to_datetime(self.df[col])
        # objects
        obj_cols = ['ship_id', 'address1', 'customer_id', 'sales_rep_id', 'item_id', 'old_item_id',
                    'item_UPC', 'old_item_UPC', 'ship_list_pk', 'sales_rep_id_2', 'list_header_id']
        for col in obj_cols:
            self.df[col] = self.df[col].astype(object)

    def _zip_code_int(self, row):
        """
        Make zip code an integer value
        Input:
            row - dataframe row (passed by apply function)
        Returns:
            First 5 digits of zip code as integer
        """
        index = self.df.columns.get_loc('postal_code')
        code = row[index]
        return int(code[:5])

    def _days_between_visits(self):
        """
        Determine the number of days between this visit and the last visit (from prev_visit_date column)
        Returns:
            List of floats representing the days between visits for every row in the data set
        """
        out_arr = []
        i = 0
        for index, row in self.df.iterrows():
            diff = row['visit_date'] - row['prev_visit_date']
            out_arr.append(pd.Timedelta(diff).days)
        return out_arr

    def create(self):
        """
        Creates a number of columns
        """
        self.df['zip_code'] = self.df.apply(self._zip_code_int, axis=1)

        # normalize target variables
        days_list = self._days_between_visits()
        self.df['qty_shrink_per_day'] = self.df.qty_shrink / days_list
        self.df['shrink_value_per_day'] = self.df.shrink_value / days_list

    def drop(self):
        """
        Drops duplicates and columns that aren't needed
        """
        del self.df['address3'] # redundant info (same as address 2)
        del self.df['postal_code'] # create zip code
        del self.df['duration'] # all zero values
        del self.df['dist_customer_id'] # all -1 values
        del self.df['POG_version_timestamp'] # dup of visit_date

    def dummy(self):
        """
        Creates dummy variables (one hot encondes) for specified columns
        """
        dummy_cols = ['item_category', 'customer_id']
        foo = pd.DataFrame()
        foo[dummy_cols] = self.df[dummy_cols].astype(str)
        self.df = pd.get_dummies(self.df, columns=dummy_cols)
        self.df[dummy_cols] = foo[dummy_cols]
        del foo

    def nans(self):
        """
        Drops rows that contain a nan value
        """
        self.df.dropna(axis=0, inplace=True)

    def fit(self, df, y=None):
        """
        Placeholder fit method required by sklearn
        """
        return self

    def transform(self, df):
        """
        Wrapper to execute data cleaning functions above
        Input:
            df - Dataframe
        Returns:
            Cleaned dataframe
        """
        self.df = df.copy()
        self.data_type()
        self.create()
        self.drop()
        self.dummy()
        if self.remove_nan_rows:
            self.nans()
        return self.df

if __name__ == "__main__":
    pass