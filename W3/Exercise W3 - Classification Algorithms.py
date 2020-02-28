from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

# Our analysis will be organized as follows:
# 1. Preliminary data analysis
# 2. K Nearest Neighbour
# 3. Support Vector Classification
# 4. Gaussian Naive Bayes

# 1. Preliminary data analysis
# For this exercise we are going to use the Iris plants dataset:
iris = datasets.load_iris()
type(iris)
# Dataset is of type Bunch, as we learned in this week's exercises

print(iris.keys())
print(iris.DESCR)
print('The three classes of the Iris plant are: ', iris.target_names)
print(iris.data.shape)

# To leverage on the pandas functionality we transform the data to DataFrames
df_X = pd.DataFrame(iris.data, columns=iris.feature_names)
df_y = pd.DataFrame(iris.target, columns=['Class'])

# sns.set(style="darkgrid")
# ax = sns.countplot(x="Class", data=df_y)

# Some summary statistics:
df_y.info()
df_y.describe()

df_X.head()
df_X.describe()
df_X.info()

fig1, ax1 = plt.subplots(nrows=2, ncols=2, constrained_layout=False, figsize=(7, 8))
fig1.suptitle('Distribution of plant features', fontsize=16)
for i, feature in enumerate(df_X.columns):
    r, c = int(i/2), i % 2
    ax1[r, c].hist(df_X.loc[df_y['Class'] == 0, feature], bins=10, color='red', alpha=0.5)
    ax1[r, c].hist(df_X.loc[df_y['Class'] == 1, feature], bins=10, color='blue', alpha=0.5)
    ax1[r, c].hist(df_X.loc[df_y['Class'] == 2, feature], bins=10, color='green', alpha=0.5)
    ax1[r, c].grid(axis='both', linestyle='dotted')
    ax1[r, c].set_xlabel(feature)
    # ax1[r, c].set_title(feature)
    if c == 0:
        ax1[r, c].set_ylabel('Frequency')
fig1.legend(iris.target_names, bbox_to_anchor=(0.77, 0.10), ncol=3)
fig1.subplots_adjust(bottom=0.18)


# Sepal sizes
plt.figure(figsize=(5, 5))
plt.scatter(df_X.loc[df_y['Class'] == 0, 'sepal length (cm)'], df_X.loc[df_y['Class'] == 0, 'sepal width (cm)'],
            color='red', alpha=0.5)
plt.scatter(df_X.loc[df_y['Class'] == 1, 'sepal length (cm)'], df_X.loc[df_y['Class'] == 1, 'sepal width (cm)'],
            color='blue', alpha=0.5)
plt.scatter(df_X.loc[df_y['Class'] == 2, 'sepal length (cm)'], df_X.loc[df_y['Class'] == 2, 'sepal width (cm)'],
            color='green', alpha=0.5)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Sepal sizes of Irises', fontsize=16)
plt.legend(iris.target_names, ncol=1)
plt.grid(axis='both', linestyle='dotted')

# Petal sizes
plt.figure(figsize=(5, 5))
plt.scatter(df_X.loc[df_y['Class'] == 0, 'petal length (cm)'], df_X.loc[df_y['Class'] == 0, 'petal width (cm)'],
            color='red', alpha=0.5)
plt.scatter(df_X.loc[df_y['Class'] == 1, 'petal length (cm)'], df_X.loc[df_y['Class'] == 1, 'petal width (cm)'],
            color='blue', alpha=0.5)
plt.scatter(df_X.loc[df_y['Class'] == 2, 'petal length (cm)'], df_X.loc[df_y['Class'] == 2, 'petal width (cm)'],
            color='green', alpha=0.5)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Petal sizes of Irises', fontsize=16)
plt.legend(iris.target_names)
plt.grid(axis='both', linestyle='dotted')


# 2. K Nearest Neighbour
# split our sample
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.50, random_state=10)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train, y_train.values.ravel())
y_pred_knn = knn.predict(X_test)

# Print the accuracy based on our test set
print('Accuracy: ', knn.score(X_test, y_test))

# print('Accuracy: ', accuracy_score(y_test, y_pred_knn))
# print('Precision: ', precision_score(y_test, y_pred_knn, average='micro'))
# print('Recall: ', recall_score(y_test, y_pred_knn, average='micro'))

# Similar to the lessons we are going to look for the best amount of neighbours, looping from 1 until 40
neighbors = np.arange(1, 40)
train_acc = np.empty(len(neighbors))
test_acc = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    knn_loop = KNeighborsClassifier(n_neighbors=k)
    knn_loop.fit(X_train, y_train.values.ravel())
    train_acc[i] = knn_loop.score(X_train, y_train)
    test_acc[i] = knn_loop.score(X_test, y_test)

# Generate plot
plt.figure(figsize=(5, 5))
plt.title('Accuracy k-NN')
plt.plot(neighbors, train_acc, label='Train')
plt.plot(neighbors, test_acc, label='Test')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

# This suggests that around k=12 neighbours would give us the best performance

# 3. Supporting Vector Classification (SVC)
# For our following models we are only going to investigate the classification of the Iris type based on Petal length
X_train_petal = X_train.iloc[:, 2:4].copy()
X_test_petal = X_test.iloc[:, 2:4].copy()

# We are going to explore two Supporting Vector Classivication models with diffferent kernels
# For this exercise we will set the regularization parameter at C=1 for both models
C = 1.0
svc_lin = svm.SVC(kernel='linear', C=C).fit(X_train_petal, y_train.values.ravel())
y_pred_svc_lin = svc_lin.predict(X_test_petal)
print('Accuracy with linear kernel: ', svc_lin.score(X_test_petal, y_test))

svc_rbf = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train_petal, y_train.values.ravel())
y_pred_svc_rbf = svc_rbf.predict(X_test_petal)
print('Accuracy with rbf kernel: ', svc_rbf.score(X_test_petal, y_test))

svc_poly = svm.SVC(kernel='poly', degree=5, C=C).fit(X_train_petal, y_train.values.ravel())
y_pred_svc_poly = svc_poly.predict(X_test_petal)
print('Accuracy with poly kernel: ', svc_poly.score(X_test_petal, y_test))

# 4. Gaussian Naive bayes
gnb = GaussianNB()
gnb.fit(X_train_petal, y_train.values.ravel())
print('Accuracy: ', gnb.score(X_test_petal, y_test))

points = 100
ls_X = np.linspace(min(X_train_petal.iloc[:, 0]), max(X_train_petal.iloc[:, 0]), points)
ls_Y = np.linspace(min(X_train_petal.iloc[:, 1]), max(X_train_petal.iloc[:, 1]), points)
X_grid, Y_grid = np.meshgrid(ls_X, ls_Y)

ls_Z = np.array([gnb.predict([[x_coord, y_coord]])[0] for x_coord, y_coord in zip(np.ravel(X_grid), np.ravel(Y_grid))])
Z_grid = ls_Z.reshape(X_grid.shape)

# Make the plot again, but fill in the regions
plt.figure(figsize=(5, 5))
plt.scatter(X_train_petal.loc[df_y['Class'] == 0, 'petal length (cm)'], X_train_petal.loc[df_y['Class'] == 0, 'petal width (cm)'],
            color='red', alpha=0.5)
plt.scatter(X_train_petal.loc[df_y['Class'] == 1, 'petal length (cm)'], X_train_petal.loc[df_y['Class'] == 1, 'petal width (cm)'],
            color='blue', alpha=0.5)
plt.scatter(X_train_petal.loc[df_y['Class'] == 2, 'petal length (cm)'], X_train_petal.loc[df_y['Class'] == 2, 'petal width (cm)'],
            color='green', alpha=0.5)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Petal sizes of Irises', fontsize=16)
plt.legend(iris.target_names)
plt.grid(axis='both', linestyle='dotted')
plt.contourf(X_grid, Y_grid, Z_grid, 2, alpha=0.15, colors=('red', 'blue', 'green'))
plt.contour(X_grid, Y_grid, Z_grid, 2, alpha=0.8, colors=('red', 'blue', 'green'))

