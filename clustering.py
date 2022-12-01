from sklearn import cluster
import logging
import matplotlib.pyplot as plt
from matplotlib import colors
import json
import datetime
import os


def clustering(algo, y, X, file):
    #cmap = plt.get_cmap('jet')
    #norm = colors.BoundaryNorm(np.arange(-2.5, 3, 1), cmap.N)
    rightnow = str(datetime.datetime.today()).replace(
        " ", "_").replace(":", "-")[:-7]
    folder = f'./clustering/{rightnow}_{file}_{algo}'
    os.makedirs(folder)
    
    print(f'clustering start, algo={algo}')
    with open(f'./cfg/ml_model/{algo}_config.json') as f:
        algo_cfg = json.load(f)

    if(algo == "KMeans"):
        #pca = decomposition.PCA(n_components=2)
        #pca_X = pca.fit_transform(X)
        # print(pca_X)

        kmeans = cluster.KMeans(n_clusters=algo_cfg['n_clusters'], init=algo_cfg['init'], n_init=algo_cfg['n_init'], max_iter=algo_cfg['max_iter'],
                                tol=algo_cfg['tol'], random_state=algo_cfg['random_state'], copy_x=algo_cfg['copy_x'], algorithm=algo_cfg['algorithm']).fit(X)
        clustering_res = kmeans.labels_

        # plt.figure(12)
        # for i in set(y):
        #    plt.scatter(pca_X[:, 0][np.where(y == i)[0]],
        #                pca_X[:, 1][np.where(y == i)[0]],  cmap=cmap, norm=norm, label=i)
        # plt.legend()
        #plt.colorbar(ticks=np.linspace(0, 1, 2))
        # plt.savefig(f'./clustering/{file}-{algo}.png')
    elif(algo == "DBSCAN"):

        dbscan = cluster.DBSCAN(eps=algo_cfg['eps'], min_samples=algo_cfg['min_samples'], metric=algo_cfg['metric'], metric_params=algo_cfg['metric_params'],
                                algorithm=algo_cfg['algorithm'], leaf_size=algo_cfg['leaf_size'], p=algo_cfg['p'], n_jobs=algo_cfg['n_jobs'],).fit(X)
        clustering_res = dbscan.labels_

    data = X
    data[y.name] = y
    data[algo] = clustering_res
    data.to_csv(f'./{folder}/{file}-{algo}.csv')
    print("clustering finish!")
