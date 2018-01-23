#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

__author__ = "Luke Olson"

class PublicData(BaseEstimator, TransformerMixin):
    """
    A generic class
    """

    def __init__(self, include_crime=False, remove_nan_rows=True):
        """
        Constructor
        """
        self.include_crime = include_crime
        self.remove_nan_rows = remove_nan_rows

    def fit(self, df, y=None):
        return self

    def _load_data(self):
        self.fd = pd.read_pickle('data/Food_Deserts/FD_clean.pkl').set_index('Zip Code')
        self.unemp = pd.read_pickle('data/Unemployment/unemp_clean.pkl').set_index('Zip')
        #self.inc = pd.read_pickle('data/Income/income_clean.pkl').set_index('ZIPCODE')
        self.dens = pd.read_pickle('data/Pop_Density/density_clean.pkl').set_index('Zip/ZCTA')
        if self.include_crime:
            self.crime = pd.read_pickle('data/Crime/grouped_clean.pkl').set_index(['state', 'city'])

    def _join(self):
        self.df = self.df.join(self.fd, on=['zip_code'], how='left')
        self.df = self.df.join(self.unemp, on=['zip_code'], how='left')
        # df = df.join(self.inc, on=['zip_code']  , how='left')
        self.df = self.df.join(self.dens, on=['zip_code'], how='left')
        if self.include_crime:
            self.df = self.df.join(self.crime, on=['state', 'city'], how='left')

        self.df['dens_sq_mile'] = self.df['dens/sq_mile'].replace(0, np.nan)
        del self.df['dens/sq_mile']


    def transform(self, df):
        self.df = df.copy()
        self._load_data()
        self._join()
        # drop all rows that contain nan
        if self.remove_nan_rows:
            self.df.dropna(axis=0, inplace=True)
            
        return self.df

if __name__ == "__main__":
    pass