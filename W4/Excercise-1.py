from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


pd.set_option('display.max_columns', None)

tracks = pd.read_csv('W4/datasets/data/fma-rock-vs-hiphop.csv', sep=',')
echonest_metrics = pd.read_json('W4/datasets/data/echonest-metrics.json', precise_float=True)
echo_tracks = pd.merge(echonest_metrics, tracks.loc[:, ['track_id', 'genre_top']], on='track_id')

# https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html

# Create a correlation matrix
corr_metrics = echo_tracks.corr()
corr_metrics.style.background_gradient()

features = echo_tracks.drop(columns=['track_id', 'genre_top'])
labels = echo_tracks.loc[:, 'genre_top']
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(features)

pca = PCA()
pca.fit(scaled_train_features)
exp_variance = pca.explained_variance_ratio_

fig, ax = plt.subplots()
ax.bar(range(pca.n_components_), exp_variance)
ax.set_xlabel('Principal Component #')

cum_exp_variance = np.cumsum(exp_variance)
fig, ax = plt.subplots()
ax.plot(range(pca.n_components_), cum_exp_variance)
ax.axhline(y=0.9, linestyle='--')
n_components = 6

pca = PCA(n_components, random_state=10)
pca.fit(scaled_train_features)
pca_projection = pca.transform(scaled_train_features)

train_features, test_features, train_labels, test_labels = train_test_split(pca_projection, labels, random_state=10)
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_features, train_labels)
pred_labels_tree = tree.predict(test_features)

logit = LogisticRegression(random_state=10)
logit.fit(train_features, train_labels)
pred_labels_logit = logit.predict(test_features)

class_rep_tree = classification_report(test_labels, pred_labels_tree)
class_rep_log = classification_report(test_labels, pred_labels_logit)

print("Decision Tree: \n", class_rep_tree)
print("Logistic Regression: \n", class_rep_log)


# Balance the dataset:
hop_only = echo_tracks.loc[echo_tracks["genre_top"] == "Hip-Hop"]
rock_only = echo_tracks.loc[echo_tracks["genre_top"] == "Rock"]
rock_only = rock_only.sample(len(hop_only), random_state=10)
rock_hop_bal = pd.concat([rock_only, hop_only])
features = rock_hop_bal.drop(['genre_top', 'track_id'], axis=1)
labels = rock_hop_bal['genre_top']
pca_projection = pca.fit_transform(scaler.fit_transform(features))
train_features, test_features, train_labels, test_labels = train_test_split(pca_projection, labels, random_state=10)

# Train our decision tree on the balanced data
tree_b = DecisionTreeClassifier(random_state=10)
tree_b.fit(train_features, train_labels)
pred_labels_tree_b = tree_b.predict(test_features)
logit_b = LogisticRegression(random_state=10)
logit_b.fit(train_features, train_labels)
pred_labels_logit_b = logit_b.predict(test_features)

# Compare the models
print("Decision Tree: \n", classification_report(test_labels, pred_labels_tree_b))
print("Logistic Regression: \n", classification_report(test_labels, pred_labels_logit_b))

kfold = KFold(n_splits=10, random_state=10, shuffle=True)
tree = DecisionTreeClassifier(random_state=10)
logit = LogisticRegression(random_state=10)
tree_score = cross_val_score(tree, pca_projection, labels, cv=kfold)
logit_score = cross_val_score(logit, pca_projection, labels, cv=kfold)
print("Decision Tree:", np.mean(tree_score), "Logistic Regression:", np.mean(logit_score))
