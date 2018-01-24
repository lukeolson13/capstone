#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from model_functions import clust_grid

__author__ = "Luke Olson"

class PredModel(BaseEstimator, TransformerMixin):
    """
    A generic class
    """

    def __init__(self, model, param_grid):
        """
        Constructor
        """
        self.model = model
        self.param_grid = param_grid

    def grid(self):
        self.best_params_list = clust_grid(self.model, self.param_grid, self.X, self.y, self.model_mask_cols)

    def create_models(self):
        self.model_list = []
        for i in range(0, len(self.X.cluster.unique())):
            foo_model = self.model
            foo_model.set_params(**self.best_params_list[i])
            self.model_list.append(foo_model)

    def fit(self, X, y):
        self.X = X
        self.y = y
        numb_no_time_mask = (self.X.dtypes == int) | (self.X.dtypes == np.float64) | (self.X.dtypes == np.uint8)
        self.model_mask_cols = self.X.columns[numb_no_time_mask]
        self.grid()
        self.create_models()
        for index, model in enumerate(self.model_list):
            fit_clust_mask = self.X.cluster == str(index)
            model.fit(self.X[ self.model_mask_cols ][fit_clust_mask], self.y[fit_clust_mask])
        return self.model_list

    def predict(self, X_pred):
        y_pred = [np.nan] * len(X_pred)
        for index, model in enumerate(self.model_list):
            pred_clust_mask = X_pred.cluster == str(index)
            y_pred_clust = model.predict(X_pred[ self.model_mask_cols ][pred_clust_mask])
            mask_indices = np.where(pred_clust_mask == True)[0]
            for place, index in enumerate(mask_indices):
                y_pred[index] = y_pred_clust[place]
        return y_pred

if __name__ == "__main__":
    pass