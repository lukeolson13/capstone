#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import time

__author__ = "Luke Olson"

class DataClean(BaseEstimator, TransformerMixin):
    """
    A generic class
    """

    def __init__(self, remove_nan_rows=True):
        """
        Constructor
        """
        self.remove_nan_rows = remove_nan_rows

    def _date_to_int(self, row, col):
        index = self.df.columns.get_loc(col)
        date = row[index]
        return time.mktime(time.strptime(str(date), '%Y-%m-%d %H:%M:%S'))

    def data_type(self):
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

    def _zip_code_inc(self, row):
        index = self.df.columns.get_loc('postal_code')
        code = row[index]
        return int(code[:5])

    def _days_between_visits(self):
        out_arr = []
        i = 0
        for index, row in self.df.iterrows():
            diff = row['visit_date'] - row['prev_visit_date']
            out_arr.append(pd.Timedelta(diff).days)
        return out_arr

    def create(self):
        self.df['zip_code'] = self.df.apply(self._zip_code_inc, axis=1)

        # normalize target variables
        days_list = self._days_between_visits()
        self.df['qty_shrink_per_day'] = self.df.qty_shrink / days_list
        self.df['shrink_value_per_day'] = self.df.shrink_value / days_list

    def drop(self):
        del self.df['address3'] # redundant info (same as address 2)
        del self.df['postal_code'] # create zip code
        del self.df['duration'] # all zero values
        del self.df['dist_customer_id'] # all -1 values
        del self.df['POG_version_timestamp'] # dup of visit_date

    def dummy(self):
        dummy_cols = ['item_category', 'customer_id']
        foo = pd.DataFrame()
        foo[dummy_cols] = self.df[dummy_cols].astype(str)
        self.df = pd.get_dummies(self.df, columns=dummy_cols)
        self.df[dummy_cols] = foo[dummy_cols]
        del foo

    def nans(self):
        self.df.dropna(axis=0, inplace=True)

    def fit(self, df, y=None):
        return self

    def transform(self, df):
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