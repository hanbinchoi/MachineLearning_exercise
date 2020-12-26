import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
X = pd.read_csv("mouse.csv",header=None)
X.columns=['x1','x2']

# DBSCAN

eps = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
min_samples = [3,5,10,15,20,30,50,100]
for e in eps:
    for min in min_samples:

        #Set Model
        model = DBSCAN(eps=e, min_samples=min)
        model.fit(X)
        X["group"] = model.labels_

        # Visualize
        grouped = X.groupby('group')
        fig, ax = plt.subplots()

        for cluster, group in grouped:
            # 클러스터가 -1이면 noise
            if cluster == -1:
                ax.plot(group.x1, group.x2, marker='o', linestyle='', color='black')
            else:
                ax.plot(group.x1, group.x2, marker='o', linestyle='', label="cluster" + str(cluster))
        ax.legend(loc='upper left')
        plt.title("eps = "+str(e)+", min_samples = "+str(min))
        plt.show()

        # 데이터셋 초기화
        X = X.drop("group", axis=1)

# K-means

cluster = [2,3,4,5,6]
max_iter = [50, 100, 200, 300]
for c in cluster:
    for max in max_iter:
        # k-means
        model = KMeans(n_clusters=c, max_iter=max)
        model.fit(X)
        X["group"] = model.labels_

        # Visualize
        grouped = X.groupby('group')
        fig, ax = plt.subplots()
        for cluster, group in grouped:
            # 클러스터 내에 데이터 개수가 50개 이하면 노이즈로 판단
            if grouped.size()[cluster] < 50:
                ax.plot(group.x1, group.x2, marker='o', linestyle='', label="noise", color='black')
            else:
                ax.plot(group.x1, group.x2, marker='o', linestyle='', label="cluster" + str(cluster))
        ax.legend(loc='upper left')
        plt.title("n_clusters = " + str(c) + ", max_iter = " + str(max))
        plt.show()

        # 데이터셋 초기화
        X = X.drop("group", axis=1)

# EM Clustering

components = [2,3,4,5,6]
max_iter = [50, 100, 200, 300]
for c in components:
    for max in max_iter:
        # k-means
        model = GaussianMixture(n_components=c, max_iter=max)
        model = model.fit(X)
        labels = model.predict(X)
        X["group"] = labels

        # Visualize
        grouped = X.groupby('group')
        fig, ax = plt.subplots()
        for cluster, group in grouped:
            # 클러스터 내에 데이터 개수가 50개 이하면 노이즈로 판단
            if grouped.size()[cluster] < 50:
                ax.plot(group.x1, group.x2, marker='o', linestyle='', label="noise", color='black')
            else:
                ax.plot(group.x1, group.x2, marker='o', linestyle='', label="cluster" + str(cluster))
        ax.legend(loc='upper left')
        plt.title("n_componets = " + str(c) + ", max_iter = " + str(max))
        plt.show()

        # 데이터셋 초기화
        X = X.drop("group", axis=1)


