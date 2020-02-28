from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import W4.utils as utils
import seaborn as sns

xs = utils.points[:, 0]
ys = utils.points[:, 1]
plt.figure(figsize=(5, 4))
plt.scatter(xs, ys)
plt.show()

model = KMeans(n_clusters=3)
model.fit(utils.points)
labels = model.predict(utils.new_points)


# Assign the columns of new_points: xs and ys
xs = utils.new_points[:, 0]
ys = utils.new_points[:, 1]

# Assign the cluster centers: centroids
centroids = model.cluster_centers_
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]

plt.figure(figsize=(5, 4))
plt.scatter(xs, ys, alpha=0.5, c=labels)
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(utils.points)

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.figure(figsize=(5, 4))
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# Three is the best number

# IRIS DATASET

iris = datasets.load_iris()
df_X = pd.DataFrame(iris.data, columns=iris.feature_names)
df_y = pd.DataFrame(iris.target, columns=['Class'])
# samples = df_X.iloc[:, [0, 2]] # Only sepald length and petal length
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_X)
labels = kmeans.predict(df_X)
xs = df_X.iloc[:, 0]
ys = df_X.iloc[:, 2]
plt.figure(figsize=(5, 4))
plt.scatter(xs, ys, c=labels)

#Crosstable
pd.crosstab(labels, df_y.values.ravel())

k_inert = []
ks_iris = range(1, 6)
for k in ks_iris:
    kmeans_loop = KMeans(n_clusters=k)
    kmeans_loop.fit(df_X)
    k_inert.append(kmeans_loop.inertia_)

# Plot ks vs inertias
plt.figure(figsize=(5, 4))
plt.plot(ks_iris, k_inert, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks_iris)
plt.show()


# StandardScaler
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
pipeline = make_pipeline(scaler, kmeans)
# pipeline.fit(df_X)
labels_pipeline = pipeline.fit_predict(df_X)
pd.crosstab(labels_pipeline, df_y.values.ravel())


normalizer = Normalizer()
kmeans = KMeans(n_clusters=3)
pipeline = make_pipeline(normalizer, kmeans)
labels2 = pipeline.fit_predict(df_X)
pd.crosstab(labels2, df_y.values.ravel())
