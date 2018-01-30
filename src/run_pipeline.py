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
	print('Segmenting Customers...')
	pub = PublicData(include_crime=False, remove_nan_rows=True)
	df_public = pub.fit_transform(df)

	cs = CustSeg(clusters=num_clusts, plot_clusts=plot_clusts, plot_sil=plot_sil)
	cust_table = cs.fit_transform(df_public)
	return cust_table

def run_pred_model(df, non_feature_cols, cust_table, grid_search=True, model=None, param_grid=None, user_model_list=None, rmse_plot=False):
	print('Running Prediction Model...')
	print('Creating lag variables...')
	cl = CreateLag(lag_periods=2, col_filters=['address1', 'item_category'], 
                   date_col='visit_date', lag_vars=['qty_shrink_per_day', 'shrink_value_per_day'], 
                   col_name_suf='_by_cat', remove_nan_rows=True)
	df_lag = cl.fit_transform(df)

	print('Adding public data...')
	pub = PublicData(include_crime=False, remove_nan_rows=True)
	df_public = pub.fit_transform(df_lag)

	print('Splitting...')
	spl = Split(non_feature_cols, target_col='shrink_value_per_day', split_by_time=False)
	spl.fit(df_public, cust_table)
	X, y, X_train, X_test, y_train, y_test = spl.transform(df_public, cust_table)

	print('Standardizing...')
	ss = StdScale(std=True, scale=True)
	X_train_ss = ss.fit_transform(X_train)
	X_test_ss = ss.fit_transform(X_test)

	print('Fitting model...')
	pm = PredModel(grid_search=grid_search, model=model, param_grid=param_grid, user_model_list=user_model_list)
	fitted_models = pm.fit(X_train_ss, y_train)

	if rmse_plot:
		print('Plotting...')
		pred_col_mask = (X_train_ss.dtypes == int) | (X_train_ss.dtypes == np.float64) | (X_train_ss.dtypes == np.uint8)
		pred_cols = X_train_ss.columns[pred_col_mask]
		cluster_rmse, naive_rmse = model_clusters(fitted_models, X_test=X_test_ss, X_test_ns=X_test, naive_col='shrink_value_per_day_lag1_by_cat', col_mask=pred_cols, y_test=y_test)
		plot_rmse(cluster_rmse, naive_rmse, num_clusters=len(X.cluster.unique()), 
          title='Predicting Next Visit Shrink Value')
	return pm, X_train_ss, y_train

def run_forc_model(df, non_feature_cols, cust_table, grid_search=True, model=None, param_grid=None, user_model_list=None, rmse_plot=True):
	print('Running Forecast Model...')
	print('Creating lag variables...')
	cl = CreateLag(lag_periods=2, col_filters=['address1'], 
                   date_col='visit_date', lag_vars=['qty_shrink_per_day', 'shrink_value_per_day'], 
                   col_name_suf='_by_store', remove_nan_rows=True)
	df_lag = cl.fit_transform(df)

	print('Adding public data...')
	pub = PublicData(include_crime=False, remove_nan_rows=True)
	df_public = pub.fit_transform(df_lag)

	print('Splitting...')
	split_date = pd.to_datetime('12/1/2017')
	spl = Split(non_feature_cols, target_col='shrink_value_per_day', split_by_time=True, date_col='visit_date', split_date=split_date)
	spl.fit(df_public, cust_table)
	X, y, X_train, X_test, y_train, y_test = spl.transform(df_public, cust_table)

	print('Standardizing...')
	ss = StdScale(std=True, scale=True)
	X_ss = ss.fit_transform(X)
	X_train_ss = ss.fit_transform(X_train)
	X_test_ss = ss.fit_transform(X_test)

	print('Fitting model...')
	forc_cols = ['FD_ratio', 'LAPOP1_10', 'POP2010', 'dens_sq_mile', 'unemp_rate', 'qty_POG_limit', 'unit_price', 'shrink_value_per_day_lag1_by_store', 'shrink_value_per_day_lag2_by_store' ]
	for col in X_ss.columns:
	    if 'customer_id' in col:
	        forc_cols.append(col)
	fc = Forecast(forc_cols, grid_search=grid_search, model=model, param_grid=param_grid, user_model_list=user_model_list, num_periods=4)
	fitted_models = fc.fit(X_ss, y)
	if rmse_plot:
		print('Plotting...')
		# plot forecast against current training data
		cluster_rmse, naive_rmse = model_clusters(fitted_models, X_test=X_test_ss, X_test_ns=X_test, naive_col='shrink_value_per_day_lag1_by_store', col_mask=forc_cols, y_test=y_test)
		plot_rmse(cluster_rmse, naive_rmse, num_clusters=len(X.cluster.unique()), 
          title='Training Forecast')
		# plot forecast against future data
		forc_model_test(X_test, y_test, fitted_models, col_mask=forc_cols, max_periods=10)
	return fc.forecast(cust_table)

if __name__ == '__main__':
	print('Reading and cleaning data...')
	df = pd.read_pickle('../data/SRP/raw_subset_300k.pkl')
	dc = DataClean(remove_nan_rows=True)
	df = dc.fit_transform(df)
	
	# customer table and segmentation
	load_table = True
	if load_table:
		try:
			cust_table = pd.read_pickle('../data/SRP/cust_table_out.pkl')
		except:
			cust_table = run_cust_seg(df, num_clusts=4, plot_clusts=True, plot_sil=False)
	else:
		cust_table = run_cust_seg(df, num_clusts=4, plot_clusts=True, plot_sil=False)
	print(cust_table.groupby('cluster').mean())

	# build models
	non_feature_cols = ['shrink_value', 'shrink_to_sales_value_pct', 'shrink_value_out', 'shrink_to_sales_value_pct_out', 'shrink_value_ex_del', 'shrink_to_sales_value_pct_ex_del', 'qty_inv_out', 'qty_shrink', 'qty_shrink_ex_del', 'qty_shrink_out', 'qty_end_inventory', 'qty_f', 'qty_out', 'qty_ex_del', 'qty_n', 'qty_delivery', 'qty_o', 'qty_d', 'qty_shrink_per_day', 'shrink_value_per_day', 'qty_start_inventory']
	use_MLP=True
	if use_MLP:
		model = MLPRegressor()
		params = {'alpha': [0.0001, 0.001, 0.01, 1], 'hidden_layer_sizes': [(300,), (100,), (50,50), (50,50,50)], 'learning_rate_init': [0.01, 0.001, 0.0001], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['adam'],  'max_iter': [3000]}
		params_basic = {'hidden_layer_sizes': [(100,)], 'learning_rate_init': [0.001], 
	          'activation': ['relu'], 'solver': ['adam'],  'max_iter': [100]}
	else:
		model = RandomForestRegressor()
		params = {'n_estimators': [20, 100, 300], 'max_features': [3, 6, 9, 12, 15], 'max_depth': [None, 10, 30], 'min_samples_leaf': [1, 5, 10], 'max_leaf_nodes': [None, 20, 100],  'n_jobs': [-1]}
		params_basic = {'n_estimators': [10], 'n_jobs': [-1]}

	# daily prediction model
	#pm, X_train_ss, y_train = run_pred_model(df, non_feature_cols, cust_table, grid_search=True, model=model, param_grid=params, user_model_list=None, rmse_plot=True)

	# forecast model
	cust_table_agg = run_forc_model(df, non_feature_cols, cust_table, grid_search=True, model=model, param_grid=params, user_model_list=None, rmse_plot=True)