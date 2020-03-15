from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import plot_importance
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from scipy.stats import uniform
from sklearn import svm
import seaborn as sns
import pandas as pd
import numpy as np
import os

pd.set_option('display.max_columns', None)
# plt.rcParams['figure.figsize'] = [20, 10]
sns.set(style="darkgrid")

# Read in the data
census = pd.read_csv('ml-challenge-week6/census-income.data',
                     sep=' *, *',
                     header=None,
                     na_values=['?', 'Do not know'],
                     engine='python')

colnames_read = pd.read_csv('ml-challenge-week6/census-income.names',
                            sep=' *: *',
                            header=None,
                            skiprows=141,
                            usecols=[0], engine='python')
colnames = colnames_read.drop([24]).append(['income'], ignore_index=True).values.ravel()
census.columns = colnames
census = census.drop(columns='instance weight')
census['income'].value_counts()


def cv_kaggle(model, x, y, cv, state):
    """
    This function performs a cross-validation of a specific model and returns an array of roc_auc scores.
    The roc_auc values are calculated based on the predicted values and not the predicted probabilities.
    It splits the sample in a train and validation set for a specified (cv) number of times. Each sample will
    be in a validation set exactly once.

    :param model: statistical model to be validated
    :param x: DataFrame with features
    :param y: DataFrame with target variable
    :param cv: integer specifying the number of k-fold cross validation
    :param state:
    :return: array with all the roc_auc scores of each split
    """

    # First shuffle the dataset, because the input dataset might be sorted (for instance, as a result of balancing)
    reshuffle_all = x.join(y).sample(frac=1, random_state=state).reset_index(drop=True)
    x = reshuffle_all.iloc[:, :-1]
    y = reshuffle_all.iloc[:, -1]

    result = [None] * cv
    max_row = x.shape[0] - 1
    low = 0
    high = round(max_row / cv)
    for i in range(cv):
        test = range(low, high)
        x_loop_train, x_loop_test, y_loop_train, y_loop_test = x.drop(test), x.iloc[test], y.drop(test), y.iloc[test]
        model.fit(x_loop_train, y_loop_train.values.ravel())
        y_loop_pred = np.round(model.predict(x_loop_test))
        result[i] = roc_auc_score(y_loop_test, y_loop_pred)
        low = high
        high = round(max_row / cv * (i + 2))
    return result


enc_income = LabelEncoder()
census.loc[:, ['income']] = enc_income.fit_transform(census['income'])
X = census.drop(columns=['income'])
y = pd.DataFrame(census['income'])

drop_columns = ['migration code-change in msa',  # Too many missing
                'migration code-change in reg',  # Too many missing
                'migration code-move within reg',  # Too many missing
                'migration prev res in sunbelt',  # Too many missing
                'detailed occupation recode',  # Already used on grouped level
                'detailed industry recode',  # Already used on grouped level
                'enroll in edu inst last wk',  # Too many in "Not in universe"
                'member of a labor union',  # Too many in "Not in Universe"
                'reason for unemployment',  # Too many in "Not in Universe"
                'region of previous residence',  # Too many in "Not in Universe"
                'state of previous residence',  # Too many in "Not in Universe"
                'detailed household and family stat',  # Too many categories/detail, we use higher level summary
                'family members under 18',  # Too many in "Not in Universe"
                "fill inc questionnaire for veteran's admin"]  # Too many in "Not in Universe"

X = X.drop(columns=drop_columns)


# get the modes for the ones that we are going to fill
mode_hisp = X['hispanic origin'].value_counts().index[0]
mode_birth_self = X['country of birth self'].value_counts().index[0]
mode_birth_mom = X['country of birth mother'].value_counts().index[0]
mode_birth_dad = X['country of birth father'].value_counts().index[0]


def fill_na_cats(x):
    x.loc[:, ['hispanic origin']] = x['hispanic origin'].fillna(mode_hisp)
    x.loc[:, ['country of birth self']] = x['country of birth self'].fillna(mode_birth_self)
    x.loc[:, ['country of birth mother']] = x['country of birth mother'].fillna(mode_birth_mom)
    x.loc[:, ['country of birth father']] = x['country of birth father'].fillna(mode_birth_dad)
    return x


class DefaultZero(dict):
    def __missing__(self, key):
        return 0


def group_cats(x):
    x['education'] = x['education'] \
        .replace(to_replace='.*grade.*', value='Still in high school', regex=True) \
        .replace(to_replace='.*Associates degree.*', value='Some college', regex=True) \
        .replace(to_replace='Some college but no degree', value='Some college', regex=True)

    map_race = {
        'White': 1,
        'Black': 0,
        'Asian or Pacific Islander': 0,
        'Other': 0,
        'Amer Indian Aleut or Eskimo': 0
    }

    x['race'] = x['race'].map(map_race)

    map_hisp = {
        'All other': 0,
        'Mexican (Mexicano)': 1,
        'Mexican-American': 1,
        'Puerto Rican': 1,
        'Central or South American': 1,
        'Other Spanish': 1,
        'Chicano': 1,
        'Cuban': 1
    }
    x['hispanic origin'] = x['hispanic origin'].map(map_hisp)

    map_birth = DefaultZero({'United-States': 1})
    x['country of birth self'] = x['country of birth self'].map(map_birth)
    x['country of birth father'] = x['country of birth father'].map(map_birth)
    x['country of birth mother'] = x['country of birth mother'].map(map_birth)
    return x


X = fill_na_cats(X)
X = group_cats(X)

var_num = ['age',
           'wage per hour',
           'capital gains',
           'capital losses',
           'dividends from stocks',
           'weeks worked in year']

var_cat = X.drop(columns=var_num).columns.values.tolist()

NumImputer = ColumnTransformer([
    ("gains_imputer", SimpleImputer(missing_values=99999, strategy='median'), ['capital gains'])
], remainder=SimpleImputer(strategy='median'))

num_pipeline = Pipeline([
    ('imputer', NumImputer),
    ('std_scaler', StandardScaler()),
])

var_cat_label = ['sex',
                 'race',
                 'hispanic origin',
                 'country of birth self',
                 'country of birth father',
                 'country of birth mother',
                 'year']

var_cat_onehot = X.drop(columns=var_num).drop(columns=var_cat_label).columns.values

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, var_num),
    ("cat", OneHotEncoder(sparse=False), var_cat_onehot),
    ("sex", OneHotEncoder(drop='first', sparse=False), var_cat_label)
])

X = pd.DataFrame(full_pipeline.fit_transform(X))


def univariate_test(x_data, y_data, score, k):
    fs = SelectKBest(score_func=score, k=k)
    fs.fit(x_data, y_data)
    df_scores = pd.DataFrame(fs.scores_)
    df_pvalues = pd.DataFrame(fs.pvalues_)
    df_columns = pd.DataFrame(x_data.columns)
    df_fs = pd.concat([df_columns, df_scores, df_pvalues], axis=1)
    df_fs.columns = ['Specs', 'Score', 'p-value']
    df_fs_sort = df_fs.sort_values('Score', ascending=False).reset_index(drop=True)
    return df_fs_sort


fs_num = univariate_test(X.iloc[:, 0:6], y.values.ravel(), score=f_classif, k='all')
fs_cat = univariate_test(X.iloc[:, 6:], y.values.ravel(), score=chi2, k='all')

sns.catplot(x='p-value',
            y='Specs',
            data=fs_cat[100:],
            kind="bar",
            color='steelblue',
            height=5,
            aspect=3)

drop = fs_cat.loc[101:, 'Specs'].values

X = X.drop(columns=drop)

# I tried PCA to select the most important features, but for every feature I removed, the worse our model performed
# (with respect to our kaggle performance cv)
# pca = PCA()
# pca.fit_transform(X)
# features = range(pca.n_components_)
# plt.bar(features, pca.explained_variance_)
# X = pd.DataFrame(pca.transform(X))


# Model fitting

# Fit a model

# Balanced sample

y['income'].value_counts()
Xy = X.join(y)
over_50 = Xy.loc[Xy["income"] == 1]
under_50 = Xy.loc[Xy["income"] == 0]
under_50_new = under_50.sample(len(over_50), random_state=25)
# We should check the representativeness of the smaller sample
under_50.describe()
under_50_new.describe()
# balanced sample:
Xy_b = pd.concat([over_50, under_50_new])
X_b = Xy_b.drop(columns=['income'])
y_b = Xy_b['income']

# logit
logit = LogisticRegression(solver='liblinear', random_state=24)
logit.fit(X_b, y_b)
y_pred = logit.predict(X_b)
roc_auc_score(y_b, y_pred)

# test for bigger sample:
y_pred_all = logit.predict(X)
roc_auc_score(y, y_pred_all)

# Now, let's see our tailor made cross-validation function in action:
scores = cv_kaggle(logit, X_b, y_b, cv=5, state=24)
pd.DataFrame(scores).describe()
# I applied this model to the kaggle test set, and indeed the resulting score was around 83%.

param_grid = {'C': np.logspace(0, 10, 10),
              'penalty': ['l1', 'l2'],
              'solver': ['liblinear']
              }
logit_search = RandomizedSearchCV(
    estimator=LogisticRegression(random_state=24),
    param_distributions=param_grid,
    n_iter=10,
    scoring='roc_auc',
    cv=5,
    refit=True,
    return_train_score=True
)
logit_search.fit(X_b, y_b)
logit_search.best_estimator_.get_params()
logit_best = logit_search.best_estimator_
pred_logit_best = logit_search.predict(X_b)
roc_auc_score(y_b, pred_logit_best)
scores = cv_kaggle(logit_best, X_b, y_b, cv=5, state=24)
pd.DataFrame(scores).describe()


# Regression tree
tree_reg = DecisionTreeRegressor(random_state=25)
tree_reg.fit(X_b, y_b)
y_pred_tree = tree_reg.predict(X_b)
roc_auc_score(y_b, y_pred_tree)
# cross-validation
scores = cv_kaggle(tree_reg, X_b, y_b, cv=5, state=24)
pd.DataFrame(scores).describe()

# Random forest
forest_reg = RandomForestRegressor(random_state=25)
forest_reg.fit(X_b, y_b)
y_pred_forest = forest_reg.predict(X_b)
roc_auc_score(y_b, np.round(y_pred_forest))

# cross-validation
scores = cv_kaggle(forest_reg, X_b, y_b, cv=5, state=24)
pd.DataFrame(scores).describe()

# Forest classifier
forest_class = RandomForestClassifier(random_state=25)
forest_class.fit(X_b, y_b)
y_pred_forest_c = forest_class.predict(X_b)
roc_auc_score(y_b, np.round(y_pred_forest_c))

# cross-validation
scores = cv_kaggle(forest_class, X_b, y_b, cv=5, state=24)
pd.DataFrame(scores).describe()

# GNB
gnb = GaussianNB()
gnb.fit(X_b, y_b)
y_pred_gnb = gnb.predict(X_b)
roc_auc_score(y_b, np.round(y_pred_gnb))

# cross-validation
scores = cv_kaggle(gnb, X_b, y_b, cv=5, state=24)
pd.DataFrame(scores).describe()


# GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(X_b, y_b)
y_pred_gb = gb.predict(X_b)
roc_auc_score(y_b, np.round(y_pred_gb))

# cross-validation
scores = cv_kaggle(gb, X_b, y_b, cv=5, state=24)
pd.DataFrame(scores).describe()


# SVC
svc = svm.SVC()
svc.fit(X_b, y_b)
y_pred_svc = svc.predict(X_b)
roc_auc_score(y_b, np.round(y_pred_svc))

# cross-validation
scores = cv_kaggle(svc, X_b, y_b, cv=5, state=24)
pd.DataFrame(scores).describe()

# k nearest neighbour
knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(X_b, y_b)
y_pred_knn = knn.predict(X_b)
roc_auc_score(y_b, np.round(y_pred_knn))

# cross-validation
scores = cv_kaggle(knn, X_b, y_b, cv=5, state=24)
pd.DataFrame(scores).describe()

# XGB Classifier
xgb = XGBClassifier()
xgb.fit(X_b, y_b)
y_pred_xgb = xgb.predict(X_b)
roc_auc_score(y_b, y_pred_xgb)
# xgb.feature_importances_
# plot_importance(xgb)

scores = cv_kaggle(xgb, X_b, y_b, cv=5, state=24)
pd.DataFrame(scores).describe()


# score_dict = xgb.get_booster().get_score()
# df_scores = pd.DataFrame(list(score_dict.items()))
# df_scores.columns = ['feature', 'score']
# df_scores.sort_values('score')
#
# drop_columns_xgb = df_scores.sort_values('score').iloc[:19].loc[:, 'feature'].values
# X_b = X_b.drop(columns=drop_columns_xgb)


# KAGGLE
#  Apply the model to the kaggle test set:
kaggle_test = pd.read_csv('ml-challenge-week6/census-income.test',
                          sep=' *, *',
                          header=None,
                          na_values=['?', 'Do not know'],
                          engine='python',
                          names=colnames[0:-1]
                          )

kaggle_clean = kaggle_test.drop(columns=drop_columns)
kaggle_clean = fill_na_cats(kaggle_clean)
kaggle_clean = group_cats(kaggle_clean)
kaggle_clean = pd.DataFrame(full_pipeline.transform(kaggle_clean))
kaggle_clean = kaggle_clean.drop(columns=drop)

# Predict:
# Unfortunately there seems to be some sort of "bug" in XGBClassifier, which saves numerical columns as strings
# with additional whitespaces. So we get an error if we predict with new data. We solve this by manually renaming
# the feature names.
feature_names = [i.strip() for i in xgb.get_booster().feature_names]
xgb.get_booster().feature_names = feature_names

kaggle_pred = xgb.predict(kaggle_clean)
kaggle_test['income class'] = np.round(kaggle_pred)
submission = kaggle_test['income class']
submission.to_csv('W6/submission.csv', sep=',', index_label='index')

os.system('kaggle competitions submit -c ml-challenge-week6 -f W6/submission.csv -m "XGB"')
