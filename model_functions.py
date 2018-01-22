import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def model_clusters(model_list, X_train, X_test, X_test_ns, naive_col, col_mask, y_train, y_test):
    if len(model_list) != len(X_train.cluster.unique()):
        print('Model list does not match number of clusters')
        return
    cluster_rmse = []
    naive_rmse = []
    for index, model in enumerate(model_list):
        print('cluster: ',index + 1)
        train_clust_mask = X_train.cluster == str(index)
        test_clust_mask = X_test.cluster == str(index)
        model.fit(X_train[col_mask][train_clust_mask], y_train[train_clust_mask])
        y_pred = model.predict(X_test[col_mask][test_clust_mask])
        y_naive = X_test_ns[test_clust_mask][naive_col].values
        cluster_rmse.append(np.sqrt(mean_squared_error(y_test[test_clust_mask], y_pred)))
        naive_rmse.append(np.sqrt(mean_squared_error(y_test[test_clust_mask], y_naive)))
        if model.__class__.__name__ == 'RandomForestRegressor':
            print()
            argsort = np.argsort(-model.feature_importances_)[:5]
            print(col_mask[argsort])
            print()
    return cluster_rmse, naive_rmse, model_list

def avg_dec(cluster_rmse, naive_rmse):
    avg_dec = []
    for i in range(0, len(cluster_rmse)):
        if (naive_rmse[i] == 0) & (cluster_rmse[i] == 0):
            continue
        avg_dec.append((naive_rmse[i] - cluster_rmse[i]) / naive_rmse[i])
    return np.mean(avg_dec)

def plot_rmse(cluster_rmse, naive_rmse, num_clusters, title='Root-Mean-Square Error'):
    fig = plt.figure(figsize=(8,8))
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
    plt.show()

def clust_grid(model, params, X_train, y_train, mask_cols):
    for clust in range(0, len(X_train.cluster.unique())):
        print()
        print('cluster: ', clust + 1)
        test_model = model
        train_clust_mask = X_train.cluster == str(clust)
        grid = GridSearchCV(test_model, param_grid=params, verbose=0)
        grid.fit(X_train[mask_cols][train_clust_mask], y_train[train_clust_mask])
        print(grid.best_params_)
        print()