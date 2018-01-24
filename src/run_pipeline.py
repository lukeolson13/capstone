import pandas as pd
from data_clean import DataClean
from add_public_data import PublicData
from create_lag import CreateLag
from cust_seg import CustSeg
from split import Split
from std_scale import StdScale
from pred_model import PredModel
from forcast import Forcast

def run_cust_seg(df):
	print('Segmenting Customers...')
	pub = PublicData(include_crime=False, remove_nan_rows=True)
	df_public = pub.fit_transform(df)

	cs = CustSeg(clusters=4, plot=True)
	cust_table = cs.fit_transform(df_public)
	return cust_table

def run_pred_model(df, non_feature_cols, cust_table, model, params):
	print('Running Prediction Model...')
	cl = CreateLag(lag_periods=2, col_filters=['address1', 'item_category'], 
                   date_col='visit_date', lag_vars=['qty_shrink_per_day', 'shrink_value_per_day'], 
                   col_name_suf='_by_cat', remove_nan_rows=True)
	df_lag = cl.fit_transform(df)

	pub = PublicData(include_crime=False, remove_nan_rows=True)
	df_public = pub.fit_transform(df_lag)

	spl = Split(non_feature_cols, target_col='shrink_value_per_day', split_by_time=False)
	spl.fit(df_public, cust_table)
	X, y, X_train, X_test, y_train, y_test = spl.transform(df_public, cust_table)

	ss = StdScale(std=True, scale=True)
	X_train_ss = ss.fit_transform(X_train)
	X_test_ss = ss.fit_transform(X_test)

	pm = PredModel(model, params)
	pm.fit(X_train_ss, y_train)
	return pm

def run_forc_model(df, non_feature_cols, cust_table, model, params):
	print('Running Forcast Model...')
	cl = CreateLag(lag_periods=2, col_filters=['address1'], 
                   date_col='visit_date', lag_vars=['qty_shrink_per_day', 'shrink_value_per_day'], 
                   col_name_suf='_by_store', remove_nan_rows=True)
	df_lag = cl.fit_transform(df)

	pub = PublicData(include_crime=False, remove_nan_rows=True)
	df_public = pub.fit_transform(df_lag)

	split_date = pd.to_datetime('12/15/2017')
	spl = Split(non_feature_cols, target_col='shrink_value_per_day', split_by_time=True, date_col='visit_date', split_date=split_date)
	spl.fit(df_public, cust_table)
	X, y, X_train, X_test, y_train, y_test = spl.transform(df_public, cust_table)

	ss = StdScale(std=True, scale=True)
	X_ss = ss.fit_transform(X)
	X_train_ss = ss.fit_transform(X_train)
	X_test_ss = ss.fit_transform(X_test)

	forc_cols = ['FD_ratio', 'LAPOP1_10', 'POP2010', 'dens_sq_mile', 'unemp_rate', 'qty_POG_limit', 'unit_price', 'shrink_value_per_day_lag1_by_store', 'shrink_value_per_day_lag2_by_store' ]
	for col in X_ss.columns:
	    if 'customer_id' in col:
	        forc_cols.append(col)
	fc = Forcast(model, params, forc_cols, num_periods=4)
	fc.fit(X_ss, y)
	return fc.forcast(cust_table)

if __name__ == '__main__':
	df = pd.read_pickle('data/SRP/raw_mini_subset.pkl')
	dc = DataClean(remove_nan_rows=True)
	df = dc.fit_transform(df)
	
	cust_table = run_cust_seg(df)

	non_feature_cols = ['shrink_value', 'shrink_to_sales_value_pct', 'shrink_value_out', 'shrink_to_sales_value_pct_out', 'shrink_value_ex_del', 'shrink_to_sales_value_pct_ex_del', 'qty_inv_out', 'qty_shrink', 'qty_shrink_ex_del', 'qty_shrink_out', 'qty_end_inventory', 'qty_f', 'qty_out', 'qty_ex_del', 'qty_n', 'qty_delivery', 'qty_o', 'qty_d', 'qty_shrink_per_day', 'shrink_value_per_day']
	model = MLPRegressor()
	params = {'hidden_layer_sizes': [(300,), (100,), (50,50), (50,50,50)], 'learning_rate_init': [0.01, 0.001, 0.0001], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['adam'],  'max_iter': [200]}

	pm = run_pred_model(df, non_feature_cols, cust_table, model, params)
	cust_table_agg = run_forc_model(df, non_feature_cols, cust_table, model, params)