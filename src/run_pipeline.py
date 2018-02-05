import pandas as pd
from data_clean import DataClean
from add_public_data import PublicData
from create_lag import CreateLag
from cust_seg import CustSeg
from split import Split
from std_scale import StdScale
from pred_model import PredModel
from forecast import Forecast
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
from model_functions import model_clusters, plot_rmse, forc_model_test
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt

font = {'size': 20}
rc('font', **font)
plt.style.use('seaborn-dark-palette')

def run_cust_seg(df, num_clusts=4, plot_clusts=False, plot_sil=False):
	"""
	Segments customers and builds customer table
	Inputs:
		df - dataframe
		num_clusters - number of customer clusters to segment into
		plot_clusts - plot (2-D) customer clusters
		plot_sil - plot silhouette scores
	Returns:
		customer table
	"""
	print('Segmenting Customers...')

	# add public data
	pub = PublicData(include_crime=False, remove_nan_rows=True)
	df_public = pub.fit_transform(df)

	# cluster
	cs = CustSeg(clusters=num_clusts, plot_clusts=plot_clusts, plot_sil=plot_sil)
	cust_table = cs.fit_transform(df_public)
	return cust_table

def run_pred_model(df, non_feature_cols, cust_table, grid_search=True, model=None, param_grid=None, user_model_list=None, rmse_plot=True):
	"""
	Runs prediction model
	Inputs:
		df - dataframe
		non_feature_cols - columns that are either targets or result in data leakage
		cust_table - customer table
		grid_seach - whether or not to grid search the best model parameters. If false, user_model_list must not be None
        model - model to be used in grid search. If None, user_model_list must not be None
        param_grid - parameter grid to be used in grid search. If None, user_model_list must not be None
        user_model_list - pre-determined (unfitted) models to be used. If None, grid_search, model, and param_grid must all contain values
        rmse_plot - whether or not to plot RMSE results for predictions and naive model
    Returns:
    	Fitted prediction model
    """
	print('Running Prediction Model...')

	print('Creating lag variables...')
	cl = CreateLag(lag_periods=2, col_filters=['address1', 'item_category'], 
                   date_col='visit_date', lag_vars=['qty_shrink_per_day', 'shrink_value_per_day'], 
                   col_name_suf='_by_cat', remove_nan_rows=True)
	df_lag = cl.fit_transform(df)

	print('Adding public data...')
	pub = PublicData(include_crime=False, remove_nan_rows=True)
	df_public = pub.fit_transform(df_lag)

	print('Splitting data...')
	spl = Split(non_feature_cols, target_col='qty_shrink_per_day', split_by_time=False)
	spl.fit(df_public, cust_table)
	X, y, X_train, X_test, y_train, y_test = spl.transform(df_public, cust_table)

	print('Standardizing data...')
	ss = StdScale(std=True, scale=True)
	X_train_ss = ss.fit_transform(X_train)
	X_test_ss = ss.fit_transform(X_test)

	print('Fitting prediction model...')
	pm = PredModel(grid_search=grid_search, model=model, param_grid=param_grid, user_model_list=user_model_list)
	fitted_models = pm.fit(X_train_ss, y_train)

	# plot predictions vs naive model
	if rmse_plot:
		print('Plotting...')
		pred_col_mask = (X_train_ss.dtypes == int) | (X_train_ss.dtypes == np.float64) | (X_train_ss.dtypes == np.uint8)
		pred_cols = X_train_ss.columns[pred_col_mask]
		cluster_rmse, naive_rmse = model_clusters(fitted_models, X_test=X_test_ss, X_test_ns=X_test, naive_col='shrink_value_per_day_lag1_by_cat', col_mask=pred_cols, y_test=y_test)
		plot_rmse(cluster_rmse, naive_rmse, title='Predicting Next Visit Shrink')
	
	return pm

def run_forc_model(df, non_feature_cols, cust_table, grid_search=True, model=None, param_grid=None, user_model_list=None, rmse_plot=True):
	"""
	Runs forecast model
	Inputs:
		df - dataframe
		non_feature_cols - columns that are either targets or result in data leakage
		cust_table - customer table
		grid_seach - whether or not to grid search the best model parameters. If false, user_model_list must not be None
        model - model to be used in grid search. If None, user_model_list must not be None
        param_grid - parameter grid to be used in grid search. If None, user_model_list must not be None
        user_model_list - pre-determined (unfitted) models to be used. If None, grid_search, model, and param_grid must all contain values
        rmse_plot - whether or not to plot RMSE results for predictions and naive model
    Returns:
    	Updated customer table with new forecasted values
    """
	print('Running Forecast Model...')

	print('Creating lag variables...')
	cl = CreateLag(lag_periods=2, col_filters=['address1'], 
                   date_col='visit_date', lag_vars=['qty_shrink_per_day', 'shrink_value_per_day'], 
                   col_name_suf='_by_store', remove_nan_rows=True)
	df_lag = cl.fit_transform(df)

	print('Adding public data...')
	pub = PublicData(include_crime=False, remove_nan_rows=True)
	df_public = pub.fit_transform(df_lag)

	print('Splitting data...')
	split_date = pd.to_datetime('12/1/2017')
	spl = Split(non_feature_cols, target_col='shrink_value_per_day', split_by_time=True, date_col='visit_date', split_date=split_date)
	spl.fit(df_public, cust_table)
	X, y, X_train, X_test, y_train, y_test = spl.transform(df_public, cust_table)

	print('Standardizing data...')
	ss = StdScale(std=True, scale=True)
	X_ss = ss.fit_transform(X)
	X_train_ss = ss.fit_transform(X_train)
	X_test_ss = ss.fit_transform(X_test)

	print('Fitting forecast model...')
	forc_cols = ['FD_ratio', 'LAPOP1_10', 'POP2010', 'dens_sq_mile', 'unemp_rate', 'qty_POG_limit', 'unit_price', 'shrink_value_per_day_lag1_by_store', 'shrink_value_per_day_lag2_by_store' ]
	for col in X_ss.columns:
	    if 'customer_id' in col:
	        forc_cols.append(col)
	fc = Forecast(forc_cols, grid_search=grid_search, model=model, param_grid=param_grid, user_model_list=user_model_list, num_periods=4)
	fitted_models = fc.fit(X_ss, y)

	# plot results
	if rmse_plot:
		print('Plotting...')
		# plot forecast against current training data (similar to prediction model)
		cluster_rmse, naive_rmse = model_clusters(fitted_models, X_test=X_test_ss, X_test_ns=X_test, naive_col='shrink_value_per_day_lag1_by_store', col_mask=forc_cols, y_test=y_test)
		plot_rmse(cluster_rmse, naive_rmse, title='Training Forecast')
		# plot forecast against future data (using predictions as next visit lag value)
		forc_model_test(X_test, y_test, fitted_models, col_mask=forc_cols, max_periods=10)
	# return customer table with forecasted shrink values
	return fc.forecast(cust_table)

if __name__ == '__main__':
	# review most important user input fields
	df_file = '../data/SRP/raw_subset_20k.pkl'
	load_table = True
	use_MLP=True
	use_basic_params = True

	# read and clean data
	print('Reading and cleaning data...')
	df = pd.read_pickle(df_file)
	dc = DataClean(remove_nan_rows=True)
	df = dc.fit_transform(df)
	
	# customer table and segmentation
	if load_table:
		# use previous determined customer clusters
		try:
			cust_table = pd.read_pickle('../data/SRP/cust_table_out.pkl')
		except:
			cust_table = run_cust_seg(df, num_clusts=4, plot_clusts=False, plot_sil=False)
	else:
		# create new customer clusters
		cust_table = run_cust_seg(df, num_clusts=4, plot_clusts=False, plot_sil=False)
	# view average values for clusters to get an idea of how they were clustered
	print(cust_table.groupby('cluster').mean())

	# build models
	non_feature_cols = ['shrink_value', 'shrink_to_sales_value_pct', 'shrink_value_out', 'shrink_to_sales_value_pct_out', 'shrink_value_ex_del', 'shrink_to_sales_value_pct_ex_del', 'qty_inv_out', 'qty_shrink', 'qty_shrink_ex_del', 'qty_shrink_out', 'qty_end_inventory', 'qty_f', 'qty_out', 'qty_ex_del', 'qty_n', 'qty_delivery', 'qty_o', 'qty_d', 'qty_shrink_per_day', 'shrink_value_per_day', 'qty_start_inventory']
	if use_MLP:
		# use multilayer perceptron
		model = MLPRegressor()
		params_grid = {'alpha': [0.0001, 0.001, 0.01, 1], 'hidden_layer_sizes': [(300,), (100,), (50,50), (50,50,50)], 'learning_rate_init': [0.01, 0.001, 0.0001], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['adam'],  'max_iter': [3000]}
		params_basic = {'hidden_layer_sizes': [(100,)], 'learning_rate_init': [0.001], 
	          'activation': ['relu'], 'solver': ['adam'],  'max_iter': [100]}
	else:
		# use a random forest
		model = RandomForestRegressor()
		params_grid = {'n_estimators': [20, 100, 300], 'max_features': [3, 6, 9, 12, 15], 'max_depth': [None, 10, 30], 'min_samples_leaf': [1, 5, 10], 'max_leaf_nodes': [None, 20, 100],  'n_jobs': [-1]}
		params_basic = {'n_estimators': [10], 'n_jobs': [-1]}

	if use_basic_params:
		params = params_basic
	else:
		params = params_grid

	# daily prediction model
	pm = run_pred_model(df, non_feature_cols, cust_table, grid_search=True, model=model, param_grid=params, user_model_list=None, rmse_plot=True)
	# can now call pm.predict on any new data

	# forecast model
	cust_table_agg = run_forc_model(df, non_feature_cols, cust_table, grid_search=True, model=model, param_grid=params, user_model_list=None, rmse_plot=True)
	# can now perform aggregate functions on new customer table to forecast shrink for regions