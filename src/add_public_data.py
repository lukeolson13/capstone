#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

__author__ = "Luke Olson"

class PublicData(BaseEstimator, TransformerMixin):
    """
    Take public data sources and joins them with the current pandas dataframe.
    """

    def __init__(self, include_crime=False, remove_nan_rows=True):
        """
        Initializer
        Inputs:
            include_crime - whether or not to include crime data given that it will rid of ~2/3 of data
            remove_nan_rows - whether or not to remove any rows from the dataframe that contain a nan value
        """
        self.include_crime = include_crime
        self.remove_nan_rows = remove_nan_rows

    def _load_data(self):
        """
        Loads public data as new dataframes
        """
        self.fd = pd.read_pickle('../data/Food_Deserts/FD_clean.pkl').set_index('Zip Code')
        self.unemp = pd.read_pickle('../data/Unemployment/unemp_clean.pkl').set_index('Zip')
        #self.inc = pd.read_pickle('../data/Income/income_clean.pkl').set_index('ZIPCODE')
        self.dens = pd.read_pickle('../data/Pop_Density/density_clean.pkl').set_index('Zip/ZCTA')
        if self.include_crime:
            self.crime = pd.read_pickle('../data/Crime/grouped_clean.pkl').set_index(['state', 'city'])

    def _join(self):
        """
        Joins public dataframes with input dataframe
        """
        self.df = self.df.join(self.fd, on=['zip_code'], how='left')
        self.df = self.df.join(self.unemp, on=['zip_code'], how='left')
        # df = df.join(self.inc, on=['zip_code']  , how='left')
        self.df = self.df.join(self.dens, on=['zip_code'], how='left')
        if self.include_crime:
            self.df = self.df.join(self.crime, on=['state', 'city'], how='left')

        self.df['dens_sq_mile'] = self.df['dens/sq_mile'].replace(0, np.nan)
        del self.df['dens/sq_mile']

    def _zip_code_str(self, row):
        """
        Converts 5 or 9 digit zip code into 5 digit string
        Input:
            row - dataframe row (passed by apply function)
        Returns:
            5 digit zip code as string
        """
        index = self.df.columns.get_loc('zip_code')
        code = row[index]
        return str(code).zfill(5)

    def fit(self, df, y=None):
        """
        Placeholder fit method required by sklearn
        """
        return self

    def transform(self, df):
        """
        Takes dataframe and adds public data
        Input:
            df - dataframe
        Returns:
            dataframe with new public data columns
        """
        self.df = df.copy()
        self._load_data()
        self._join()
        # drop all rows that contain nan
        if self.remove_nan_rows:
            self.df.dropna(axis=0, inplace=True)
        self.df['zip_code'] = self.df.apply(self._zip_code_str, axis=1)
            
        return self.df

if __name__ == "__main__":
    pass