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

def avg_dec(cluster_rmse, naive_rmse):
    avg_dec = []
    for i in range(0, len(cluster_rmse)):
        if (naive_rmse[i] == 0) & (cluster_rmse[i] == 0):
            continue
        avg_dec.append((naive_rmse[i] - cluster_rmse[i]) / naive_rmse[i])
    return np.mean(avg_dec)

def plot_rmse(cluster_rmse, naive_rmse, num_clusters, title='Root-Mean-Square Error'):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    clusters = np.arange(0, num_clusters)
    ax.bar(x=clusters, height=cluster_rmse, width=0.4, label='Model')
    ax.bar(x=np.arange(0.4, num_clusters + 0.4), height=naive_rmse, width=0.4, label='Naive')
    ax.set_xticks(clusters)
    ax.set_xticklabels(clusters)
    ax.set_xlabel('Cluster #')
    ax.set_ylabel('Root-Mean-Square Error ($/day/item)')
    ax.set_title(title)
    ax.grid(alpha=0.4)
    ax.legend()
    print('Average decrease: {}%'.format(round(avg_dec(cluster_rmse, naive_rmse), 3) * 100))
    plt.savefig('../images/{}.png'.format(title.replace(" ", "")))
    #plt.show()

def clust_grid(model, params, X_train, y_train, mask_cols):
    best_params_list = []
    for clust in range(0, len(X_train.cluster.unique())):
        print()
        print('cluster: ', clust + 1)
        test_model = model
        train_clust_mask = X_train.cluster == str(clust)
        grid = GridSearchCV(test_model, param_grid=params, verbose=0)
        grid.fit(X_train[mask_cols][train_clust_mask], y_train[train_clust_mask])
        best_params = grid.best_params_
        print(best_params)
        best_params_list.append(best_params)
    return best_params_list

def class_crossval_plot(X, y, models, scoring='neg_mean_absolute_error'):
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

    fig = plt.figure(figsize=(25, 18))
    plt.tight_layout()
    fig.suptitle('Algorithm Comparison of CrossVal Scores')
    ax = fig.add_subplot(111)
    sb.violinplot(data=results, orient='v')
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('K-Fold CV Negative Mean Abs. Error')
    ax.set_xlabel('Model')
    plt.grid(alpha=0.4)
    plt.savefig('../images/model_selection.png')

def _split_and_plot(rmse_dict):
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
        plot_rmse(cluster_rmse, naive_rmse, num_clusts, title='{} Time Period(s) Forward'.format(int(i) + 1))

def forc_model_test(X_test, y_test, test_cluster_models, col_mask):
    rmse_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
    lag1_loc = X_test[col_mask].columns.get_loc('shrink_value_per_day_lag1_by_store')
    lag2_loc = X_test[col_mask].columns.get_loc('shrink_value_per_day_lag2_by_store')
    for add in X_test.address1.unique():
        add_mask = X_test.address1 == add
        foo = X_test[ add_mask ].sort_values('visit_date', ascending=True)
        clust = int(foo.cluster.values[0])
        # set initial lag variables to current value
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

            lag2_val = lag1_val
            lag1_val = pred
                                
            i += 1
    _split_and_plot(rmse_dict)

def pred_shrink_value(cust_table, start_date, end_date, num_periods):
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