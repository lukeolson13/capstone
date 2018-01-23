from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from matplotlib import rc

font = {'size': 20}
rc('font', **font)
plt.style.use('seaborn-bright')

def kmeans(X_km, clusters):
    SSE_arr = []
    ss_arr = []
    for i in clusters:
        kmeans = KMeans(n_clusters=i, n_jobs=-1)
        clust_dist = kmeans.fit_transform(X_km)
        clust_num = kmeans.predict(X_km)

        SSE = 0
        for a, b in zip(clust_dist, clust_num):
            SSE += a[b] ** 2
        SSE_arr.append(SSE)

        if i > 1:
            ss_arr.append(silhouette_score(X_km, clust_num))
    return SSE_arr, ss_arr

def elbow_plot(clusters, SSE_arr):
    plt.figure(figsize=(12,8))
    plt.title('Elbow Plot')
    plt.plot(clusters, SSE_arr)
    plt.grid(alpha=0.3)
    plt.xticks(clusters)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squares Error (SSE)')
    plt.savefig('images/elbow.png')
    plt.show()

def silhouette_plot(clusters, ss_arr):
    plt.figure(figsize=(12,8))
    plt.title('Silhouette Scores')
    plt.plot(clusters, ss_arr)
    plt.grid(alpha=0.3)
    plt.xticks(clusters)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig('images/silhouette.png')
    plt.show()

def heir_clust(X_hc, thresh, dist_metric='cosine', num_params_to_display=50):
    # Find distances using pair-wise distances in the array, according to desired metric
    dist = squareform(pdist(X_hc.values.T, metric = dist_metric))
    # Plot dendrogram
    fig, axarr = plt.subplots(nrows = 3, ncols = 1, figsize=(60, 80))
    for ax, linkmethod in zip(axarr.flatten(), ['single', 'complete', 'average']):
        clust = linkage(dist, method=linkmethod)
        dendrogram(clust, ax=ax, truncate_mode='lastp', p=num_params_to_display, labels=model_mask_cols,
                   color_threshold=thresh, leaf_font_size=25) #color threshold number sets the color change
        ax.set_title('{} linkage'.format(linkmethod), fontsize=40)
        ax.grid(alpha=0.3)
    plt.savefig('images/clust.png'.format(linkmethod))
    plt.show()
