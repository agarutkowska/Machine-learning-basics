# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:03:35 2020

@author: arutk
"""

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

iris = datasets.load_iris()
X=iris.data
y=iris.target

# poeksperymentować z n_clusters, max_iter, tol, verbose
# n_clusters
accuracy_scores = []
for i in range(1,11):
   kmeans = KMeans(n_clusters=i, verbose=1)
   kmeans.fit(X)
   y_pred = kmeans.predict(X)
   accuracy_scores.append(adjusted_rand_score(y, y_pred))
   
x = range(1,11)
plt.plot(x, accuracy_scores, '-b', label="Dokładnosć klasteryzacji")
plt.plot(x,accuracy_scores, '-r', label="Dokładnosć klasyfikacji")
plt.legend(loc="center right")

plt.title("Wykres zmiany w dokładności klasteryzacji \nw zależnosci od liczby klastrów.")
plt.savefig("Wykres.png")
plt.show()

# max_iter
accuracy_scores = []
for i in range(100,600,100):
   kmeans = KMeans(n_clusters=3, verbose=1, max_iter=i)
   kmeans.fit(X)
   y_pred = kmeans.predict(X)
   accuracy_scores.append(adjusted_rand_score(y, y_pred))
   
x = range(100,600,100)
plt.plot(x, accuracy_scores, '-b', label="Dokładnosć klasteryzacji")
plt.plot(x, accuracy_scores, '-r', label="Dokładnosć klasyfikacji")
plt.legend(loc="center right")

plt.title("Wykres zmiany w dokładności klasteryzacji \nw zależnosci od liczby iteracji.")
plt.savefig("Wykres2.png")
plt.show()

# Proszę wyrysować w 3D znalezione centra na wykresie z próbkami (atrybut cluster_centers_) (wizualizacja po PCA)
plt.figure('Iris dataset', figsize=(7,5))
ax = plt.axes(projection = '3d')
ax.scatter(X[:,3],X[:,0],X[:,2],c=y)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
kmeans_predicted = kmeans.predict(X)
centroids = kmeans.cluster_centers_
target_names = iris.target_names
colors = ['navy', 'turquoise', 'darkorange']

plt.figure('K-Means on Iris Dataset', figsize=(7,7))
ax = plt.axes(projection = '3d')
ax.scatter(X[:,3],X[:,0],X[:,2], c=y, cmap='Set2', s=50)

# color missclassified data
ax.scatter(X[kmeans_predicted!=y,3],X[kmeans_predicted!=y,0],X[kmeans_predicted!=y,2], c='b', s=50)

# plot centroids
ax.scatter(centroids[0,3],centroids[0,0],centroids[0,2], c='r', s=50, label='centroid')
ax.scatter(centroids[1,3],centroids[1,0],centroids[1,2], c='r', s=50)
ax.scatter(centroids[2,3],centroids[2,0],centroids[2,2], c='r', s=50)

ax.legend()

