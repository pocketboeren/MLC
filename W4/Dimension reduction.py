from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans


iris = datasets.load_iris()
df_X = pd.DataFrame(iris.data, columns=iris.feature_names)
df_y = pd.DataFrame(iris.target, columns=['Class'])

sepal_length = df_X.iloc[:, 0]
sepal_width = df_X.iloc[:, 1]
plt.scatter(sepal_length, sepal_width)
plt.axis('equal')
plt.show()
correlation, pvalue = pearsonr(sepal_length, sepal_width)
print(correlation)

pca = PCA()
pca.fit(df_X)
transformed = pca.transform(df_X)
correlation, pvalue = pearsonr(transformed[:, 0], transformed[:, 1])
print(correlation)

# decorrelated!

pca = PCA()
pca.fit(df_X)
mean = pca.mean_
# transformed = pca.transform(df_X)
features = range(pca.n_components_)
# Get the first principal component: first_pc
first_pc = pca.components_[0, :]

plt.figure(figsize=(5, 4))
plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()

plt.figure(figsize=(5, 4))
plt.scatter(df_X.iloc[:, 0], df_X.iloc[:, 1])
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)
plt.axis('equal')
plt.show()


scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler, pca)
pipeline.fit(df_X)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()


# Actual dimension reduction:
pca = PCA(n_components=2)
pca.fit(df_X)
transformed = pca.transform(df_X)
plt.figure(figsize=(5, 4))
plt.scatter(transformed[:, 0], transformed[:, 1], c=df_y.values.ravel())


documents = ['cats say meow', 'dogs say woof', 'dogs chase cats']
tfidf = TfidfVectorizer()
csr_mat = tfidf.fit_transform(documents)
print(csr_mat.toarray())
words = tfidf.get_feature_names()
print(words)

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)
kmeans = KMeans(n_clusters=6)
pipeline = make_pipeline(svd, kmeans)


