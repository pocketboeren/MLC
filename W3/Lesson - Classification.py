from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import W3.utils as ut
import seaborn as sns


plt.rcParams['figure.figsize'] = [11, 7]

ut.house_votes.head()
ut.house_votes.info()
ut.house_votes.describe()

plt.figure()
sns.countplot(x='education', hue='party', data=ut.house_votes, palette='RdBu')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# Create arrays for the features and the response variable
y = ut.house_votes['party'].values
X = ut.house_votes.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)
# Predict the labels for the training data X
y_pred = knn.predict(X)

X_new = pd.DataFrame([[0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897, 0.42310646, 0.9807642, 0.68482974,
                       0.4809319, 0.39211752, 0.34317802, 0.72904971, 0.43857224, 0.0596779, 0.39804426, 0.73799541]],
                     columns=ut.house_votes_columns[1:])
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))


digits = datasets.load_digits()
print(digits.keys())
print(digits.DESCR)
print(digits.images.shape)
print(digits.data.shape)
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


# Create feature and target arrays
X = digits.data
y = digits.target
# Split into training and test set and stratify the split according to the labels so that they are distributed in the
# training and test sets as they are in the original data set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))


# Overfitting and underfitting

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()