import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn import datasets
from sklearn.manifold import TSNE


# Hierarchical clustering
iris = datasets.load_iris()
df_X = pd.DataFrame(iris.data, columns=iris.feature_names)
df_y = pd.DataFrame(iris.target, columns=['Class'])

mergings = linkage(df_X, method='single')
labels = fcluster(mergings, 2, criterion='distance')
labels = fcluster(mergings, 3, criterion='maxclust')
pd.crosstab(labels, df_y.values.ravel())

plt.figure(figsize=(10, 10))
dendrogram(mergings, leaf_rotation=10, leaf_font_size=10)
plt.show()

labels = fcluster(mergings, 2, criterion='distance')

#t-SNE

tsne = TSNE(learning_rate=150)
transformed = tsne.fit_transform(df_X)
plt.scatter(transformed[:, 0], transformed[:, 1], c=df_y.values.ravel())
