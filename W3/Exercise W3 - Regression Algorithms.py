from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg
from sklearn import svm
import numpy as np
import statsmodels.api as sm
pd.set_option('display.max_columns', None)

# For this exercise we are going to use the Boston house prices dataset:
boston = datasets.load_boston()

# Our analysis will be organized as follows:
# 1. Preliminary data analysis
# 2. Linear regression
# 3. Ridge regression
# 4. Support Vector Regression
# 5. Regression Tree


# 1. Preliminary data analysis
type(boston)
# Dataset is of type Bunch, as we learned in this week's exercises

print(boston.keys())
print(boston.DESCR)
print(boston.data.shape)
# X, y = datasets.load_boston(return_X_y=True)

df_X = pd.DataFrame(boston.data, columns=boston.feature_names)
df_y = pd.DataFrame(boston.target, columns=['PRICE'])

# sum summary statistics and
df_y.info()
df_y.describe()

df_y.hist(bins=50)
plt.grid(axis='both', linestyle='dotted')
plt.title('Distribution of house prices in Boston')
plt.xlabel('House price (x $1,000)')
plt.ylabel('Frequency')
plt.show()


# Looks like the house price is censored from above to 50,000 USD

df_X.head()
df_X.describe()
df_X.info()


# hist_X = df_X.hist(density=1, bins=30, figsize=(10, 10))
# for r in range(hist_X.shape[0]):
#     for c in range(hist_X.shape[1]):
#         hist_X[r, c].grid(linestyle='dotted')
# plt.tight_layout()
# # plt.subplots_adjust()
# plt.suptitle('Distribution of features', size=16, va='top')
# plt.rcParams['figure.constrained_layout.use'] = True

# Pandas hist wrapper could be used for a quick histogram, but I wanted experiment in order to have more control
# df_X.hist(bins=30, figsize=(10, 10))

fig1, ax1 = plt.subplots(nrows=4, ncols=4, constrained_layout=True, figsize=(10, 10))
for i in ax1[-1, -3:]:
    i.axis('off')
fig1.suptitle('Distribution of features', fontsize=16)
for i, feature in enumerate(df_X.columns):
    r, c = int(i/4), i % 4
    ax1[r, c].hist(df_X[feature], bins=30)
    ax1[r, c].grid(axis='both', linestyle='dotted')
    ax1[r, c].set_xlabel(feature)
    # ax1[r, c].set_title(feature)
    if c == 0:
        ax1[r, c].set_ylabel('Frequency')

# The following page was useful to understand the subplot api:
# https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html


# # Create new figures in the loop
# for i, feature in enumerate(df_X.columns):
#     plt.figure(figsize=(5, 4))
#     plt.scatter(df_X[feature], df_y)
#     plt.xlabel(feature)
#     plt.ylabel('House price (x $1,000)')
#     plt.show()


fig2, ax2 = plt.subplots(nrows=4, ncols=4, constrained_layout=True, figsize=(10, 10))
for i in ax2[-1, -3:]:
    i.axis('off')
fig2.suptitle('House prices against features', fontsize=16)
for i, feature in enumerate(df_X.columns):
    r, c = int(i/4), i % 4
    ax2[r, c].scatter(df_X[feature], df_y, s=3)
    ax2[r, c].set_xlabel(feature)
    # ax2[r, c].set_title(feature)
    if c == 0:
        ax2[r, c].set_ylabel('House price (x $1,000)')

# It might be interesting to look at quadratic effects, for example for the variable RM
df_X['RM2'] = df_X['RM']**2
# df_X = df_X.drop(columns=['RM2'])
# split our sample
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# First we only look at one feature: The average number of rooms per dwelling (RM)
# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train['RM'].values.reshape(-1, 1), y_train)
ps_lin = np.linspace(min(X_train['RM']), max(X_train['RM'])).reshape(-1, 1)
plot_linear_line = linear_reg.predict(ps_lin)

# variables are just pointer to an object
# Why copy: to avoid chain indexing! Learn more:
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-view-versus-copy


# Quadratic Regression
quad_reg = LinearRegression()
quad_reg.fit(X_train[['RM', 'RM2']], y_train)
ps_quad = pd.DataFrame(ps_lin, columns=['Single'])
ps_quad['Squared'] = ps_lin**2
plot_quad_line = quad_reg.predict(ps_quad)

# plot the results
fig3, ax3 = plt.subplots()
ax3.scatter(X_train['RM'], y_train, s=3, color='grey')
ax3.plot(ps_lin, plot_linear_line, color='red', linewidth=1)
ax3.plot(ps_lin, plot_quad_line, color='blue', linewidth=1)
ax3.set_xlabel('RM (average number of rooms)')
ax3.set_ylabel('House price (x $1,000)')
fig3.suptitle('Linear and Quadratic Regression', fontsize=16)
ax3.legend(['Linear', 'Quadratic', 'Train data'])

# How well does it fit on the train data
linear_reg.score(X_train['RM'].values.reshape(-1, 1), y_train)
quad_reg.score(X_train[['RM', 'RM2']], y_train)

# But the big question: How well does it fit on the test data
score_lin = linear_reg.score(X_test['RM'].values.reshape(-1, 1), y_test)
y_pred_lin = linear_reg.predict(X_test['RM'].values.reshape(-1, 1))
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
print('For the linear model we have:')
print('R^2: {}'.format(score_lin))
print('RMSE: {}'.format(rmse_lin))

score_quad = quad_reg.score(X_test[['RM', 'RM2']], y_test)
y_pred_quad = quad_reg.predict(X_test[['RM', 'RM2']])
rmse_quad = np.sqrt(mean_squared_error(y_test, y_pred_quad))
print('For the quadratic model we have:')
print('R^2: {}'.format(score_quad))
print('RMSE: {}'.format(rmse_quad))


# Add all features
reg_all = LinearRegression()
reg_all.fit(X_train.values, y_train.values)
print(reg_all.coef_)

# Statsmodels might provide a better overview of the regression output of OLS
# For statsmodels we add the constant manually
X_train_sm = X_train.copy()
X_train_sm['Const'] = 1
# X_train_sm.assign(Const2=1)
# X_train_sm.insert(0, 'Const3', 1)
reg_all_sm = sm.OLS(y_train, X_train_sm).fit()
print(reg_all_sm.summary())

# Or manually calculate the OLS estimator beta_hat
# beta_hat = np.dot(np.linalg.inv(np.dot(X_train_sm.T, X_train_sm)), np.dot(X_train_sm.T, y_train))

# From the output we observe that some variables are not statistically significant at a 5% level
# For further analysis we might want to remove them to get a more parsimonious model
# We can consider the Ridge regression as well, wich penalizes additional parameters
# ZN, INDUS, AGE

score_all = reg_all.score(X_test, y_test)
y_pred_all = reg_all.predict(X_test)
rmse_all = np.sqrt(mean_squared_error(y_test, y_pred_all))
print('For the model with all the features we have:')
print('R^2: {}'.format(score_all))
print('RMSE: {}'.format(rmse_all))

plt.scatter(y_test, y_pred_all)
plt.xlabel("Actual price (x $1,000)")
plt.ylabel("Predicted price (x $1,000) ")
plt.title("Comparison between actual and predicted price")
plt.show()

# Ridge
# Remember from the lessen that a higher value of alpha restricts the coefficients more
# For this example we go with alpha=0.5. Normally however, we would like to do some cross validation to find the best
# value for a.
reg_ridge = Ridge(alpha=0.5)
reg_ridge.fit(X_train, y_train)
y_pred_ridge = reg_ridge.predict(X_test)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
score_ridge = reg_ridge.score(X_test, y_test)
print('For the Ridge regression with all the features we have:')
print('R^2: {}'.format(score_ridge))
print('RMSE: {}'.format(rmse_ridge))

# If we remove the features that were inconsistent
# ZN, INDUS, AGE
X_train_lm = X_train.copy().drop(['ZN', 'INDUS', 'AGE'], axis=1)
X_test_lm = X_test.copy().drop(['ZN', 'INDUS', 'AGE'], axis=1)
reg_ridge_lm = Ridge(alpha=1)
reg_ridge_lm.fit(X_train_lm, y_train)
y_pred_ridge_lm = reg_ridge_lm.predict(X_test_lm)
rmse_ridge_lm = np.sqrt(mean_squared_error(y_test, y_pred_ridge_lm))
score_ridge_lm = reg_ridge_lm.score(X_test_lm, y_test)
print('For the limited Ridge regression with less features we have:')
print('R^2: {}'.format(score_ridge_lm))
print('RMSE: {}'.format(rmse_ridge_lm))

# Even though we penalized more parameters and dropped some variables that were statistically insignificant, we still
# find that our OLS model with all features fits the test data the best, which might be evidence that we are not yet
# fitting the noise

# In the same way can explore some other regression algorithms, like Support Vector Regression or Random Forest
# SVR: Support Vector Regression
reg_svr = svm.SVR()
# There are three implementations in sklearn: SVR, NuSVR and LinearSVR. We use the first as an example.
reg_svr.fit(X_train, y_train.values.ravel())
score_svr = reg_svr.score(X_test, y_test)
y_pred_svr = reg_svr.predict(X_test)
rmse_svr_all = np.sqrt(mean_squared_error(y_test, y_pred_svr))
print('For the SVR regression model with all the features we have:')
print('R^2: {}'.format(score_svr))
print('RMSE: {}'.format(rmse_svr_all))

# Tree Regression for one variable (to produce some visuals in 2D)

reg_tree_overfit = DecisionTreeRegressor(max_depth=20)
reg_tree_overfit.fit(X_train['RM'].values.reshape(-1, 1), y_train)
y_pred_tree_overfit = reg_tree_overfit.predict(X_test['RM'].values.reshape(-1, 1))
y_pred_overfit_plot = reg_tree_overfit.predict(ps_lin)
score_tree_overfit = reg_tree_overfit.score(X_test['RM'].values.reshape(-1, 1), y_test)
rmse_tree_overfit = np.sqrt(mean_squared_error(y_test, y_pred_tree_overfit))
print('For the Decision Tree with only RM and max depth = 20, we have:')
print('R^2: {}'.format(score_tree_overfit))
print('RMSE: {}'.format(rmse_tree_overfit))

reg_tree = DecisionTreeRegressor(max_depth=3)
reg_tree.fit(X_train['RM'].values.reshape(-1, 1), y_train)
y_pred_tree = reg_tree.predict(X_test['RM'].values.reshape(-1, 1))
y_pred_plot = reg_tree.predict(ps_lin)
score_tree = reg_tree.score(X_test['RM'].values.reshape(-1, 1), y_test)
rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
print('For the Decision Tree with only RM and max depth = 2, we have:')
print('R^2: {}'.format(score_tree))
print('RMSE: {}'.format(rmse_tree))

# As can be seen, the tree with max_depth = 3 fits better on the test dataset, indicating that we are overfitting when
# we use max_depth=20

plt.figure()
plt.scatter(X_train['RM'].values, y_train, color='grey')
plt.plot(ps_lin, y_pred_plot, color='blue')
plt.plot(ps_lin, y_pred_overfit_plot, color='red')
plt.xlabel('RM (average number of rooms)')
plt.ylabel('House price (x $1,000)')
plt.title('Decision Tree Regression Results', fontsize=16)
plt.legend(['Depth = 3', 'Depth = 20', 'Train data'])

# Depth=20 clearly fits to the noise






