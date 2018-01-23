#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

__author__ = "Luke Olson"

class Forcast(BaseEstimator, TransformerMixin):
    """
    A generic class
    """

    def __init__(self):
        """
        Constructor
        """

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        return self.df

if __name__ == "__main__":
    pass