#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from model_functions import clust_grid

__author__ = "Luke Olson"

class Forecast(BaseEstimator, TransformerMixin):
    """
    Forecast shrink value for stores.
    """

    def __init__(self, model_mask_cols, grid_search=True, model=None, param_grid=None, user_model_list=None, num_periods=4):
        """
        Initializer
        Inputs:
            model_mask_cols - features columns to be used in forecast
            grid_seach - whether or not to grid search the best model parameters. If false, user_model_list must not be None
            model - model to be used in grid search. If None, user_model_list must not be None
            param_grid - parameter grid to be used in grid search. If None, user_model_list must not be None
            user_model_list - pre-determined (unfitted) models to be used. If None, grid_search, model, and param_grid must all contain values
            num_periods - number of periods forward to forecast shrink value
        """
        if grid_search & (param_grid == None):
            print('Param Grid must be passed if grid_search=True')
        elif grid_search & (model == None):
            print('Model must be passed if grid_search=True')
            return()
        elif (not grid_search) & (user_model_list == None):
            print('List of models to fit must be passed if grid_search=False')
            return
        self.model_mask_cols = model_mask_cols
        self.grid_search = grid_search
        self.model = model
        self.param_grid = param_grid
        self.user_model_list = user_model_list
        self.num_periods = num_periods

    def grid(self):
        """
        Determines best model parameters from grid search
        """
        self.best_params_list = clust_grid(self.model, self.param_grid, self.X, self.y, self.model_mask_cols)

    def _create_models(self):
        """
        Creates models from best model parameters determined by grid search
        """
        self.model_list = []
        for i in range(0, len(self.X.cluster.unique())):
            foo_model = self.model
            foo_model.set_params(**self.best_params_list[i])
            self.model_list.append(foo_model)

    def _update_cust_table(self, add, i, pred):
        """
        Update customer table with forecast values and other needed values
        Inputs:
            add - column for customer address (address1)
            i - time period
            pred - forecast shrink value prediction
        """
        self.cust_table.set_value(add, 'period{}_forc_shrink_value_per_day_per_item'.format(i), pred)
        days_i = self.cust_table.columns.get_loc('days_between_visits')
        days = self.cust_table.loc[add][days_i]
        last_visit_i = self.cust_table.columns.get_loc('last_visit')
        last_visit = self.cust_table.loc[add][last_visit_i]
        next_visit = last_visit.to_datetime()  + pd.to_timedelta(i * days, unit='D')
        self.cust_table.set_value(add, 'period{}_pred_date'.format(i), next_visit)

    def forc_model(self):
        """
        Forecast value of shrink for a time period for a specific store
        """
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
        """
        Fits forecast models
        Inputs:
            X - features to use in fitting models
            y - targets to use in scoreing models
        Returns:
            list of fitted models
        """
        self.X = X
        self.y = y
        if self.grid_search:
            # grid search over params to determine best model for each cluster
            self.grid()
            self.model_list = self._create_models()
        else:
            if len(self.X.cluster.unique()) != len(self.user_model_list):
                print('Number of models does not match number of clusters')
                return()
            else:
                # use user defined params for cluster models
                self.model_list = self.user_model_list
        for index, model in enumerate(self.model_list):
            fit_clust_mask = self.X.cluster == str(index)
            model.fit(self.X[ self.model_mask_cols ][fit_clust_mask], self.y[fit_clust_mask])
        return self.model_list

    def forecast(self, cust_table):
        """
        Forecasts shrink value
        Inputs:
            cust_table - customer table
        Returns:
            update customer table with new forecasted shrink values
        """
        self.cust_table = cust_table.copy()
        self.forc_model()
        return self.cust_table

if __name__ == "__main__":
    pass