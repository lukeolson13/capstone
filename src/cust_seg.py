#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.cluster import KMeans
from std_scale import StdScale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import rc

font = {'size': 20}
rc('font', **font)
plt.style.use('seaborn-bright')

__author__ = "Luke Olson"

class CustSeg(BaseEstimator, TransformerMixin):
    """
    A generic class
    """

    def __init__(self, clusters=4, plot=False):
        """
        Constructor
        """
        self.clusters = clusters
        self.plot = plot

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
        self.cust_table = self.df.groupby(['address1']).mean()[['qty_shrink_per_day', 'shrink_value_per_day', 'POP2010',
            'FD_ratio', 'unemp_rate', 'dens_sq_mile', ]].reset_index()
        self.cust_table.set_index('address1', inplace=True)

        city_i = self.df.columns.get_loc('city')
        state_i = self.df.columns.get_loc('state')
        zip_i = self.df.columns.get_loc('zip_code')
        cust_i = self.df.columns.get_loc('customer_id')
        for index, row in self.cust_table.iterrows():
            foo = self.df[ self.df.address1 == index]
            for i, r in foo.iterrows():
                city = r[city_i]
                state = r[state_i]
                zip_code = r[zip_i]
                cust_id = r[cust_i]

                self.cust_table.set_value(index, 'city', city)
                self.cust_table.set_value(index, 'state', state)
                self.cust_table.set_value(index, 'zip_code', zip_code)
                self.cust_table.set_value(index, 'customer_id', cust_id)
                break
        
        self._add_cols()

    def _std_cust_table(self):
        self.std_cust_table = self.cust_table.copy()
        ss = StdScale(std=True, scale=False)
        self.std_cust_table = ss.fit_transform(self.std_cust_table)

    def _cluster(self):
        shrink_cust_mask = (self.cust_table.dtypes == float)
        self.shrink_cust_cols = list(self.cust_table.columns[ shrink_cust_mask ])
        self.shrink_cust_cols.remove('avg_UPC_per_visit')
        self.shrink_cust_cols.remove('days_between_visits')
        cust_kmeans = KMeans(n_clusters=self.clusters, max_iter=10000, tol=0.00001, n_jobs=-1)
        pred = cust_kmeans.fit_predict(self.std_cust_table[self.shrink_cust_cols])
        self.cust_table['cluster'] = pred.astype(str)

    def plot_clust(self, cust_table):
        cust_pca = PCA(2)
        pcas = []
        clusts = cust_table.cluster.unique()
        for clust in clusts:
            pca = cust_pca.fit_transform(cust_table[ cust_table.cluster == clust][self.shrink_cust_cols])
            pcas.append(pca)

        plt.figure(figsize=(10,10))
        for i, pca in enumerate(pcas):
            plt.scatter(pca[:,0], pca[:,1], label=clusts[i])
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('../images/cluster.png')
        plt.show()

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        self.df = df.copy()
        self.build_cust_table()
        self._std_cust_table()
        self._cluster()
        if self.plot:
            self.plot_clust(self.cust_table)
        return self.cust_table

if __name__ == "__main__":
    pass