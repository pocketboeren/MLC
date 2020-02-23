from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# and setting the size of all plots.
plt.rcParams['figure.figsize'] = [11, 7]

gapminder = pd.read_csv('W3/data/gapminder.csv')
y = gapminder['life'].values
X = gapminder['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

y = y.reshape(-1, 1)
X = X.reshape(-1, 1)


gapminder.info()
gapminder.head()
gapminder.describe()

plt.scatter(x='fertility', y='life', data=gapminder[['fertility', 'life']])
plt.xlabel('Life Expectancy')
plt.ylabel('Fertility')
plt.show()

# Create the regressor: reg
reg = LinearRegression()
prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)
reg.fit(X, y)
y_pred = reg.predict(prediction_space)
print(reg.score(X, y))

plt.scatter(x='fertility', y='life', data=gapminder[['fertility', 'life']])
plt.xlabel('Life Expectancy')
plt.ylabel('Fertility')

plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()


# Create arrays for features and target variable
y = gapminder['life'].values
X = gapminder.drop(['life', 'Region'], axis=1).values
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))


reg = LinearRegression()
# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)
# Print the 5-fold cross-validation scores
print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

reg = LinearRegression()
cvscores_3 = cross_val_score(reg, X, y, cv=3)
cvscores_10 = cross_val_score(reg, X, y, cv=10)
print(np.mean(cvscores_3))
print(np.mean(cvscores_10))

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)
lasso.fit(X, y)
lasso_coef = lasso.coef_
print(lasso_coef)
columns = ['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP', 'BMI_female', 'child_mortality']
plt.plot(range(len(columns)), lasso_coef)
plt.xticks(range(len(columns)), columns, rotation=60)
plt.margins(0.02)
plt.show()


def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:
    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha

    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)

    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))

    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))


display_plot(ridge_scores, ridge_scores_std)


