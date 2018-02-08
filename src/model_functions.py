import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from collections import defaultdict

font = {'size': 20}
rc('font', **font)
plt.style.use('seaborn-dark-palette')

def model_clusters(model_list, X_test, X_test_ns, naive_col, col_mask, y_test):
    """
    Predict values for current model. Obtain root-mean-squared error for both model and naive model.
    Inputs:
        model_list - list of fitted models to predict values for
        X_test - scaled test features to use in prediction
        X_test_ns - non-scaled test features; uses a lag column as the naive value
        naive_col - dataframe lag column to use as last visit prediction
        col_mask - features to use in prediction model
        y_test - targets associated with test features to compare to predictions
    Returns:
        cluster_rmse - list of RMSE values, one for each input model
        naive_rmse - list of RMSE values, one for each input model
    """
    cluster_rmse = []
    naive_rmse = []
    for index, model in enumerate(model_list):
        test_clust_mask = X_test.cluster == str(index)
        y_pred = model.predict(X_test[col_mask][test_clust_mask])
        y_naive = X_test_ns[test_clust_mask][naive_col].values
        cluster_rmse.append(np.sqrt(mean_squared_error(y_test[test_clust_mask], y_pred)))
        naive_rmse.append(np.sqrt(mean_squared_error(y_test[test_clust_mask], y_naive)))
        if model.__class__.__name__ == 'RandomForestRegressor':
            print()
            print('Cluster: ', index)
            argsort = np.argsort(-model.feature_importances_)[:6]
            print('Six most important features: ', col_mask[argsort])
            print()
    return cluster_rmse, naive_rmse

def avg_dec(cluster_rmse, naive_rmse, X, yaxis_units='$/day/store'):
    """
    Determine weighted average decrease between prediction RMSE and naive RMSE for clusters
    Inputs:
        cluster_rmse - list of RMSE values for prediction model
        naive_rmse - list of RMSE values for naive model
        X - features dataframe
        yaxis_units - units of y-axis
    Returns:
        average overall decrease between models
    """
    avg_dec = []
    prop_dec = []
    # get proportion of samples in each cluster
    prop_arr = (X.groupby('cluster').count() / len(X)).FD_ratio.values

    for i in range(0, len(cluster_rmse)):
        if (naive_rmse[i] == 0) & (cluster_rmse[i] == 0):
            continue
        dec = (naive_rmse[i] - cluster_rmse[i]) / naive_rmse[i]
        avg_dec.append(dec)
        prop_dec.append(prop_arr[i])
        print('Cluster {} decrease: {}'.format(i, round(dec, 3) * 100))

    clust_avg_rmse = 0
    naive_avg_rmse = 0
    for place, prop in enumerate(prop_dec):
        clust_avg_rmse += cluster_rmse[place] * prop / sum(prop_dec)
        naive_avg_rmse += naive_rmse[place] * prop / sum(prop_dec)
    print()
    print('Average RMSE decrease ({}): '.format(yaxis_units), naive_avg_rmse - clust_avg_rmse)
    print()

    weighted_avg_dec = 0
    for dec, prop in zip(avg_dec, prop_dec):
        weighted_avg_dec += dec * prop / sum(prop_dec)
    return weighted_avg_dec

def plot_rmse(cluster_rmse, naive_rmse, X, yaxis_units='$/day/item', title='Root-Mean-Square Error'):
    """
    Visualize difference between prediction RMSE and naive RMSE
    Inputs:
        cluster_rmse - list of RMSE values for prediction model
        naive_rmse - list of RMSE values for naive model
        X - features dataframe
        yaxis_units - units of y-axis
        title - plot title
    """
    num_clusters = len(cluster_rmse)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    clusters = np.arange(0, num_clusters)
    ax.bar(x=clusters - 0.2, height=cluster_rmse, width=0.4, label='Model')
    ax.bar(x=np.arange(0.2, num_clusters + 0.2), height=naive_rmse, width=0.4, label='Naive')
    ax.set_xticks(clusters)
    ax.set_xticklabels(clusters)
    ax.set_xlabel('Cluster #')
    ax.set_ylabel('Root-Mean-Square Error ({})'.format(yaxis_units))
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    print('Average overall decrease: {}%'.format(round(avg_dec(cluster_rmse, naive_rmse, X, yaxis_units), 3) * 100))
    plt.savefig('../images/{}.png'.format(title.replace(" ", "")), bbox_inches='tight')
    plt.show()

def clust_grid(model, params, X_train, y_train, mask_cols):
    """
    Grid search over each cluster model
    Inputs:
        model - sklearn model to use (ie Lasso())
        params - parameter grid to search over for each model
        X_train - features to train model with
        y_train - targets to validate model with
        mask_cols - feature columns to use in model predictions
    Returns:
        list of the best parameters found by the grid search for each model
    """
    best_params_list = []
    for clust in range(0, len(X_train.cluster.unique())):
        print()
        print('cluster: ', clust)
        test_model = model
        train_clust_mask = X_train.cluster == str(clust)
        grid = GridSearchCV(test_model, param_grid=params, n_jobs=-1, verbose=1)
        grid.fit(X_train[mask_cols][train_clust_mask], y_train[train_clust_mask])
        best_params = grid.best_params_
        print(best_params)
        best_params_list.append(best_params)
    return best_params_list

def class_crossval_plot(X, y, models, scoring='neg_mean_absolute_error'):
    """
    Create violin plot of multiple models' test scores
    Inputs:
        X - dataframe features
        y - dataframe target column
        models - list of sklearn models to test
        scoring - measure of best fit for models to use
    """
    results = []
    names = []
    all_scores = []
    print('Mod - Avg - Std Dev')
    print('---   ---   -------')
    for name, model in models:
        kfold = KFold(n_splits=10)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring, n_jobs=-1)
        results.append(cv_results)
        names.append(name)
        print('{}: {:.2f} ({:2f})'.format(name, cv_results.mean(), cv_results.std()))

    fig = plt.figure(figsize=(16, 10))
    plt.tight_layout()
    fig.suptitle('Cross Validation Comparison of Regression Models')
    ax = fig.add_subplot(111)
    sb.violinplot(data=results, orient='v')
    ax.set_xticklabels(names, rotation=50, ha='right')
    ax.set_xlabel('Model')
    plt.grid(alpha=0.4)

def _split_and_plot(rmse_dict, X):
    """
    Plots results of forc_model_test
    Inputs:
        rmse_dict - dictionary of RMSE values passed by forc_model_test
        X - features dataframe
    """
    num_clusts = len(rmse_dict['0']['pred'])
    for i in rmse_dict.keys():
        cluster_rmse = [0] * num_clusts
        naive_rmse = [0] * num_clusts
        for kind in rmse_dict[i].keys():
            for clust in rmse_dict[i][kind].keys():
                if kind == 'pred':
                    cluster_rmse[int(clust)] = np.mean(rmse_dict[i][kind][clust])
                else:
                    naive_rmse[int(clust)] = np.mean(rmse_dict[i][kind][clust])
        plot_rmse(cluster_rmse, naive_rmse, X, yaxis_units='$/day/store', title='{} Time Period(s) Forward'.format(int(i) + 1))

def forc_model_test(X_test, y_test, test_cluster_models, col_mask, max_periods=10):
    """
    Uses predictions at each timestep to forecast shrink value forward in time, and compares with actual value
    Inputs:
        X_test - features to use in testing
        y_test - testing targets
        test_cluster_models - fitted cluster model to make predictions with
        col_mask - feature columns to use in model
        max_periods - number of time periods forward to forecast shrink value
    """
    rmse_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
    lag1_loc = X_test[col_mask].columns.get_loc('shrink_value_per_day_lag1_by_store')
    lag2_loc = X_test[col_mask].columns.get_loc('shrink_value_per_day_lag2_by_store')
    for add in X_test.address1.unique():
        add_mask = X_test.address1 == add
        foo = X_test[ add_mask ].sort_values('visit_date', ascending=True)
        clust = int(foo.cluster.values[0])
        # set naive prediction as the last visit's (lag1) value
        naive_pred = foo.shrink_value_per_day_lag1_by_store.values[0]
        lag1_val = naive_pred
        lag2_val = foo.shrink_value_per_day_lag2_by_store.values[0]
        i = 0
        for index, row in foo.iterrows():
            model = test_cluster_models[clust]
            inputs = row[col_mask].values
            inputs[lag1_loc] = lag1_val
            inputs[lag2_loc] = lag2_val
            
            pred = model.predict(inputs.reshape(1, -1))                      
            actual = [y_test[index]]
            naive = [naive_pred]

            pred_rmse = np.sqrt(mean_squared_error(actual, pred))
            naive_rmse = np.sqrt(mean_squared_error(actual, naive))
            rmse_dict['{}'.format(i)]['pred'][clust].append(pred_rmse)
            rmse_dict['{}'.format(i)]['naive'][clust].append(naive_rmse)

            # set next visit lag1 as current prediction
            lag2_val = lag1_val
            lag1_val = pred
                                
            i += 1
            if i >= max_periods:
                break
    _split_and_plot(rmse_dict, X_test)

def pred_shrink_value(cust_table, start_date, end_date, num_periods):
    """
    Forecast shrink value between two dates for each store
    Inputs:
        cust_table - customer table with prediction forecast and associated predicted visit dates
        start_date - forecast start date
        end_date - forecast end date
        num_periods - number of periods forward in time to consider shrink predictions
    Returns:
        updated customer table with new aggregate column, with single value per store
    """
    out_table = cust_table.copy()
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    day_diff = (end_date - start_date).days
    
    shrink_locs = []
    pred_date_locs = []
    for i in range(1, num_periods + 1):
        loc_shrink = cust_table.columns.get_loc('period{}_forc_shrink_value_per_day_per_item'.format(i))
        shrink_locs.append(loc_shrink)
        loc_pred_date = cust_table.columns.get_loc('period{}_pred_date'.format(i))
        pred_date_locs.append(loc_pred_date)
    for index, row in cust_table.iterrows():
        shrink_vals = []
        for place, date_loc in enumerate(pred_date_locs):
            if (row[date_loc] >= start_date) & (row[date_loc] <= end_date):
                shrink_loc = shrink_locs[place]
                shrink_vals.append(row[shrink_loc])
        agg_value = np.mean(shrink_vals) * day_diff
        out_table.set_value(index, 'agg_shrink_value', agg_value)
    return out_table
    
def flag(cust_table, mult_over_min=4, amount_over_min=100, and_or='and'):
    """
    Flags stores (prints them) that are above a certain threshold in the same zip code as another store
    Inputs:
        cust_table - customer table from pred_shrink_value
        mult_over_min - multiple over minimum shrink value for all stores in a region as threshold
        amount_over_min - value over minimum shrink value for all stores in a region as threshold
        and_or - whether both or only one threshold must be met in order to flag store
    """
    foo = cust_table[ cust_table.agg_shrink_value > 0 ] \
        .groupby(['zip_code', 'address1']).sum()[['agg_shrink_value']].reset_index()
    loc_agg = foo.columns.get_loc('agg_shrink_value')
    loc_add = foo.columns.get_loc('address1')
    for zip in foo.zip_code.unique():
        bar = foo[ foo.zip_code == zip]
        min_val = bar.agg_shrink_value.min()
        for _, row in bar.iterrows():
            shrink_val = row[loc_agg]
            val_times_over_min = shrink_val / min_val
            val_amount_over_min = shrink_val - min_val
            if and_or == 'or':
                if (val_times_over_min >= 4) | (val_amount_over_min >= amount_over_min):
                    print(row[loc_add], ' - ', val_times_over_min, ' - ', val_amount_over_min)
            else:
                if (val_times_over_min >= 4) & (val_amount_over_min >= amount_over_min):
                    print(row[loc_add], ' - ', val_times_over_min, ' - ', val_amount_over_min)