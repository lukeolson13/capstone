#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.cluster import KMeans
from std_scale import StdScale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import rc
from clust_functions import kmeans, silhouette_plot
import numpy as np

font = {'size': 20}
rc('font', **font)
plt.style.use('seaborn-dark-palette')

__author__ = "Luke Olson"

class CustSeg(BaseEstimator, TransformerMixin):
    """
    Segment customers into distinct clusters.
    """

    def __init__(self, clusters=4, plot_clusts=False, plot_sil=False):
        """
        Constructor
        """
        self.clusters = clusters
        self.plot_clusts = plot_clusts
        self.plot_sil = plot_sil

    def _last_visit(self, row):
        add = row.name
        foo = self.df[ self.df.address1 == add ].groupby(['address1', 'visit_date']).count()
        num_visits = len(foo)
        return foo.index[num_visits - 1][1]

    def _days_between_visits(self, row):
        add = row.name
        foo = self.df[ self.df.address1 == add ].groupby(['address1', 'visit_date']).count()
        num_visits = len(foo)
        first_visit = foo.index[0][1]
        last_visit = foo.index[num_visits - 1][1]
        return (last_visit - first_visit).days / num_visits

    def _add_cols(self):
        # add in avg items (UPC) per visit
        foo = self.df.groupby(['address1', 'visit_date']).count()[['qty_shrink_per_day']]
        foo['avg_UPC_per_visit'] = foo['qty_shrink_per_day']
        foo = foo.groupby('address1').mean()[['avg_UPC_per_visit']]
        self.cust_table = self.cust_table.join(foo)

        # add in avg days between visits
        self.cust_table['days_between_visits'] = self.cust_table.apply(self._days_between_visits, axis=1)

        # add in last visit date
        self.cust_table['last_visit'] = self.cust_table.apply(self._last_visit, axis=1)
    

    def build_cust_table(self):
        self.cust_table = self.df.groupby(['address1']).mean()[['qty_shrink_per_day', 'shrink_value_per_day', 'POP2010', 'FD_ratio', 'unemp_rate', 'dens_sq_mile', ]].reset_index()
        self.cust_table.set_index('address1', inplace=True)

        cust_add_table = pd.read_pickle('../data/SRP/custs_by_address.pkl')
        self.cust_table = self.cust_table.join(cust_add_table, how='left')
        try:
            del self.cust_table['latitude']
            del self.cust_table['longitude']
        except:
            pass
        
        self._add_cols()

    def _std_cust_table(self):
        self.std_cust_table = self.cust_table.copy()
        ss = StdScale(std=True, scale=False)
        self.std_cust_table = ss.fit_transform(self.std_cust_table)

    def _cluster(self):
        print('Clustering...')
        shrink_cust_mask = (self.cust_table.dtypes == float)
        self.shrink_cust_cols = list(self.cust_table.columns[ shrink_cust_mask ])
        self.shrink_cust_cols.remove('avg_UPC_per_visit')
        self.shrink_cust_cols.remove('days_between_visits')
        cust_kmeans = KMeans(n_clusters=self.clusters, max_iter=10000, tol=0.00001, n_jobs=-1)
        pred = cust_kmeans.fit_predict(self.std_cust_table[self.shrink_cust_cols])
        self.std_cust_table['cluster'] = pred.astype(str)
        self.cust_table['cluster'] = pred.astype(str)

    def plot_clust(self, cust_table):
        cust_pca = PCA(2)
        pca = cust_pca.fit_transform(cust_table[self.shrink_cust_cols])
        
        colors = ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7']
        plt.figure(figsize=(10,10))
        for clust in sorted(cust_table.cluster.unique()):
            clust_mask = cust_table.cluster == clust
            plt.scatter(pca[:,0][clust_mask], pca[:,1][clust_mask], label='Cluster {}'.format(clust), color=colors[int(clust)])
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
        plt.ylim(ymin=-9, ymax=12)
        plt.xlim(xmin=-15, xmax=15)
        plt.legend()
        plt.savefig('../images/cluster.png')
        #plt.show()

    def plot_ss(self, cust_table): 
        SSE_arr, ss_arr = kmeans(cust_table[self.shrink_cust_cols], clusters=np.arange(1, 13))
        silhouette_plot(ss_arr, clusters=np.arange(2, 13))

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        self.df = df.copy()
        self.build_cust_table()
        self._std_cust_table()
        self._cluster()
        if self.plot_clusts:
            self.plot_clust(self.std_cust_table)
        if self.plot_sil:
            self.plot_ss(self.std_cust_table)
        self.cust_table.to_pickle('../data/SRP/cust_table_out.pkl')
        return self.cust_table

if __name__ == "__main__":
    pass