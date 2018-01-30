#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from model_functions import clust_grid

__author__ = "Luke Olson"

class Forecast(BaseEstimator, TransformerMixin):
    """
    A generic class
    """

    def __init__(self, model, param_grid, model_mask_cols, num_periods=4):
        """
        Constructor
        """
        self.model = model
        self.param_grid = param_grid
        self.model_mask_cols = model_mask_cols
        self.num_periods = num_periods

    def grid(self):
        self.best_params_list = clust_grid(self.model, self.param_grid, self.X, self.y, self.model_mask_cols)

    def _create_models(self):
        self.model_list = []
        for i in range(0, len(self.X.cluster.unique())):
            foo_model = self.model
            foo_model.set_params(**self.best_params_list[i])
            self.model_list.append(foo_model)

    def _update_cust_table(self, add, i, pred):
        self.cust_table.set_value(add, 'period{}_forc_shrink_value_per_day_per_item'.format(i), pred)
        days_i = self.cust_table.columns.get_loc('days_between_visits')
        days = self.cust_table.loc[add][days_i]
        last_visit_i = self.cust_table.columns.get_loc('last_visit')
        last_visit = self.cust_table.loc[add][last_visit_i]
        next_visit = last_visit.to_datetime()  + pd.to_timedelta(i * days, unit='D')
        self.cust_table.set_value(add, 'period{}_pred_date'.format(i), next_visit)

    def forc_model(self):
        lag1_loc = self.X[self.model_mask_cols].columns.get_loc('shrink_value_per_day_lag1_by_store')
        lag2_loc = self.X[self.model_mask_cols].columns.get_loc('shrink_value_per_day_lag2_by_store')
        for add in self.X.address1.unique():
            add_mask = self.X.address1 == add
            foo = self.X[ add_mask ].sort_values('visit_date', ascending=False)
            top_index = foo.index[0]
            clust = int(foo.cluster.values[0])
            # get values from last visit for store
            base_input = foo[self.model_mask_cols].values[0]
            base_actual = self.y[top_index]
            lag2_val = base_input[lag1_loc]
            lag1_val = base_actual

            for i in range(1, self.num_periods + 1):
                model = self.model_list[clust]
                inputs = base_input
                inputs[lag1_loc] = lag1_val
                inputs[lag2_loc] = lag2_val
                
                pred = model.predict(inputs.reshape(1, -1))
                self._update_cust_table(add, i, pred)
                    
                lag2_val = lag1_val
                lag1_val = pred

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.grid()
        self._create_models()
        for index, model in enumerate(self.model_list):
            fit_clust_mask = self.X.cluster == str(index)
            model.fit(self.X[ self.model_mask_cols ][fit_clust_mask], self.y[fit_clust_mask])
        return self.model_list

    def forecast(self, cust_table):
        self.cust_table = cust_table.copy()
        self.forc_model()
        return self.cust_table

if __name__ == "__main__":
    pass