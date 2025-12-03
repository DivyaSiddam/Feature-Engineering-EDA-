#!/usr/bin/env python
# coding: utf-8

# Linear assumption
# Regression
# Linear regression is a straightforward approach for predicting a quantitative response Y on the basis of a different predictor variable X1, X2, ... Xn. It assumes that there is a linear relationship between X(s) and Y. Mathematically, we can write this linear relationship as Y ≈ β0 + β1X1 + β2X2 + ... + βnXn.
# 
# In a very simple example, X may represent number of advertisements shown on TV per day and Y may represent total number of sales of the advertised product per day. Then we can regress sales onto adds shown per day on TV by fitting the model sales ≈ β0 + β1×TV (see figure below).

# In[2]:


# simulation of a linear regression example

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

n = 50
x = np.linspace(1, 100, n)
y = x * 10 + np.random.randn(n)*80

fig, ax = plt.subplots()
fit = np.polyfit(x, y, deg=1)
ax.plot(x, fit[0] * x + fit[1], color='red')
ax.scatter(x, y)
ax.set_title('Linear Regression')
ax.set_ylabel('Number of sales per day')
ax.set_xlabel('Number of advertisements per day')


# # Classification:
# 
# Similarly, for classification, Logistic Regression assumes a linear relationship between the variables and the log of the odds.
# 
# Odds = p / 1 - p, where p is the probability of y = 1
# 
# log(odds) = β0 + β1X1 + β2X2 + ... + βnXn
# 
# In the figure below, I illustrate and explain this relationship.

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# --------------------------
# Example data
# --------------------------
# X: feature (reshape for sklearn)
X = np.linspace(-10, 10, 100).reshape(-1, 1)

# Y: labels (0 or 1) with some pattern
Y = (X.flatten() > 0).astype(int)

# --------------------------
# Fit logistic regression
# --------------------------
logit = LogisticRegression()
logit.fit(X, Y)

# Predicted probabilities
Y_prob = logit.predict_proba(X)[:, 1]

# --------------------------
# Plot
# --------------------------
fig, ax = plt.subplots(figsize=(8, 5))

# Scatter original data points
ax.scatter(X[Y == 0], Y[Y == 0], s=30, color='green', label='Class 0')
ax.scatter(X[Y == 1], Y[Y == 1], s=30, color='orange', label='Class 1')

# Plot predicted probabilities
ax.plot(X, Y_prob, color='blue', linewidth=2, label='Predicted Probability')

# Horizontal line at 0.5
ax.axhline(y=0.5, color='red', linestyle='--', label='Decision boundary')

# Labels, title, legend
ax.set_xlabel('Feature X')
ax.set_ylabel('Probability / Label')
ax.set_title('Logistic Regression')
ax.legend()
ax.grid(True)

plt.show()


# In[6]:


import matplotlib.pyplot as plt
import numpy as np

# Example data
X = np.linspace(-10, 10, 100)
Y_dots0 = np.random.rand(35)  # example values for first group
Y_dots1 = np.random.rand(35)  # example values for second group

fig, ax = plt.subplots(figsize=(8,5))

# Scatter plot for the two groups
ax.scatter(X[0:35], Y_dots0, s=3, color='green')
ax.scatter(X[65:100], Y_dots1, s=3, color='orange')

# Horizontal line at y = 0.5 (decision boundary)
ax.axhline(y=0.5, color='red')  # removed 'hold' and corrected xmin/xmax

# Optionally add dashed line across axes limits
ax.plot(ax.get_xlim(), [0.5, 0.5], ls="--", c=".3")

ax.set_title('Logistic Regression')
plt.show()


# Which algorithms assume linear relationships between predictors and outcome?
# Models that assume linear relationships between predictors and outome are:
# 
# Linear and Logistic Regression
# Linear Discriminant Analysis (LDA)
# Principal Component Regressors
# Why is it important to understand the linear assumptions?
# If the machine learning model assumes a linear dependency between the predictors Xs and the outcome Y, when there is not such a linear relationship, the model will have a poor performance. In such cases, we are better off trying another machine learning model that does not make such assumption.
# 
# Linear models are preferred in business settings for a variety of reasons:
# 
# If there is a linear relationship between Xs and Y, linear models can have very good performance
# Non-linear models like trees cannot make accurate predictions on value ranges for the target outside those of the training dataset
# Sometimes, the business wants a linear change between the output and the predictors (this would not occur with non-linear methods)
# Linear models are easier to interpret, and we can infer how each variable affects the output, thus business can comply with regulations (for example regulations to treat the customer fairly).
# 
# What can be done if there is no linear relationship?
# Sometimes a linear relationship may appear after some variable transformation. Two transformations typically used are:
# 
# Mathematical transformation of the variable.
# Discretisation.
# 
# 
# Real Life example:
# Predicting Sale Price of Houses
# The problem at hand aims to predict the final sale price of homes based on different explanatory variables describing aspects of residential homes. Predicting house prices is useful to identify fruitful investments, or to determine whether the price advertised for a house is over or underestimated, before making a buying judgment.
# 
# To download the House Price dataset go this website: kaggle
# 
# Scroll down to the bottom of the page, and click on the link 'train.csv', and then click the 'download' blue button towards the right of the screen, to download the dataset. Save it to a directory of your choice.
# 
# Note that you need to be logged in to Kaggle in order to download the datasets.
# 
# If you save it in the same directory from which you are running this notebook and name the file 'houseprice.csv' then you can load it the same way I will load it below.
# 
# ====================================================================================================
# 
# In this notebook, I will demonstrate some naturally occurring linear relationships between predictors X and output variables Y.

# In[10]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[15]:


# load the House Price Dataset, with a few columns for demonstration

cols_to_use = ['OverallQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea','WoodDeckSF',
               'BsmtUnfSF','SalePrice']

data = pd.read_csv('C:/Users/sidda/Downloads/house-prices-advanced-regression-techniques/train.csv', usecols=cols_to_use)
print(data.shape)
data.head()


# In[16]:


# plot the numerical columns vs the output SalePrice to visualise the (linear) relationship

for col in cols_to_use[:-1]:
    data.plot.scatter(x=col, y='SalePrice', ylim=(0,800000))
    plt.show()


# In[17]:


# I will group variables into those that have a somewhat linear relationship with sale price
#  and those that don't

linear_vars = ['OverallQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea']
non_linear_vars = ['WoodDeckSF', 'BsmtUnfSF']


# In[18]:


# let's separate into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    data.fillna(0), data.SalePrice, test_size=0.3, random_state=0)

X_train.shape, X_test.shape


# Assessing linear relationship: examining the errors
# One thing that we can do to determine whether there is a linear relationship between the variable and the target is:
# 
# make a linear regression model using the desired variables (X)
# 
# predict with the linear model the target
# 
# determine the error (True sale price - predicted sale price)
# 
# observe the distribution of the error.
# 
# If SalePrice is linearly explained by the variable we are evaluating, then the error should be random noise, typically following a normal distribution centered at 0. So we expect to see the error terms for each observation lying around 0.

# # OverallQual

# In[19]:


col = 'OverallQual'
linreg = LinearRegression()
linreg.fit(X_train[col].to_frame(), y_train)
print('Train set')
pred = linreg.predict(X_train[col].to_frame())
print('Linear Regression mse: {}'.format(mean_squared_error(y_train, pred)))
print('Test set')
pred = linreg.predict(X_test[col].to_frame())
print('Linear Regression mse: {}'.format(mean_squared_error(y_test, pred)))
print()
X_test['error'] = X_test.SalePrice - pred
print('Error Stats')
print(X_test['error'].describe())
X_test.plot.scatter(x=col, y='error')


# The errors should be normally distributed with a mean around 0. Tthis is not the case forthe variable OverallQual. Thus, we conclude that 'OverallQual' is not linearly related to 'SalePrice'.

# In[20]:


col = 'TotalBsmtSF'
linreg = LinearRegression()
linreg.fit(X_train[col].to_frame(), y_train)
print('Train set')
pred = linreg.predict(X_train[col].to_frame())
print('Linear Regression mse: {}'.format(mean_squared_error(y_train, pred)))
print('Test set')
pred = linreg.predict(X_test[col].to_frame())
print('Linear Regression mse: {}'.format(mean_squared_error(y_test, pred)))

X_test['error'] = X_test.SalePrice - pred
print('Error stats')
print(X_test['error'].describe())
X_test.plot.scatter(x=col, y='error', xlim=(0, 2000), ylim=(-20000, 20000))


# The mean of the model errors in this case is 288, and when compared with the average house sale price (180921), it is very small. Therefore, although SalePrice is not completely explained by the linear relationship with TotalBsmtSF, a linear model for this variable and the outcome is not a bad idea.

# # 1stFlrSF

# In[21]:


col = '1stFlrSF'
linreg = LinearRegression()
linreg.fit(X_train[col].to_frame(), y_train)
print('Train set')
pred = linreg.predict(X_train[col].to_frame())
print('Linear Regression mse: {}'.format(mean_squared_error(y_train, pred)))
print('Test set')
pred = linreg.predict(X_test[col].to_frame())
print('Linear Regression mse: {}'.format(mean_squared_error(y_test, pred)))

X_test['error'] = X_test.SalePrice - pred
print('Error stats')
print(X_test['error'].describe())
X_test.plot.scatter(x=col, y='error', xlim=(0, 2000), ylim=(-20000, 20000))


# The model error of the linear model between this variable and the outcome is small compared to house price, and althogh it does not follow a normal distribution, it looks like a linear relationship between 1stFlrSF and Sale Price is not a bad idea.

# # GrLivArea

# In[22]:


col =  'GrLivArea'
linreg = LinearRegression()
linreg.fit(X_train[col].to_frame(), y_train)
print('Train set')
pred = linreg.predict(X_train[col].to_frame())
print('Linear Regression mse: {}'.format(mean_squared_error(y_train, pred)))
print('Test set')
pred = linreg.predict(X_test[col].to_frame())
print('Linear Regression mse: {}'.format(mean_squared_error(y_test, pred)))

X_test['error'] = X_test.SalePrice - pred
print('Error stats')
print(X_test['error'].describe())
X_test.plot.scatter(x=col, y='error', xlim=(0, 3000), ylim=(-20000, 20000))


# Analysis of the errors of the linear model between this variable and Sale Price does not follow a normal distribution centered at zero. Therefore, there is not a strictly linear relationship between GrLivArea and Sale Price.

# In[25]:


# let's look at the distribution of GrLivArea
import seaborn as sns
sns.boxplot(x='GrLivArea', data=data)


# In[26]:


# let's look at the distributions of these variables

for var in cols_to_use[:-1]:
    fig = data[var].hist(bins=50)
    fig.set_xlabel(var)
    fig.set_ylabel('Number of Houses')
    plt.show()


# The first 4 variables, which show a somewhat linear relationship with the target, show as well a somewhat Gaussian distribution. The fact that the distribution is not completely Gaussian, and the relationship not completely linear, ends in the distribution of the model errors seen in the previous notebook cells.
# 
# The last 2 variables deviate from the Gaussian distribution, which in turn may affect both linear relationship with target and error distribution. See below:

# # WoodDeckSF

# In[27]:


col = 'WoodDeckSF'
linreg = LinearRegression()
linreg.fit(X_train[col].to_frame(), y_train)
print('Train set')
pred = linreg.predict(X_train[col].to_frame())
print('Linear Regression mse: {}'.format(mean_squared_error(y_train, pred)))
print('Test set')
pred = linreg.predict(X_test[col].to_frame())
print('Linear Regression mse: {}'.format(mean_squared_error(y_test, pred)))

X_test['error'] = X_test.SalePrice - pred
print('Error stats')
print(X_test['error'].describe())
X_test.plot.scatter(x=col, y='error')


# From this error plot, we can see that this variable is not linearly related to the Sale Price. Errors do not follow a normal distribution centered in zero.

# # Let's compare the performance of some machine learning models on linear variables

# In[28]:


# let's normalise the variables (this is necessary for linear regression as seen in previous lecture)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train[linear_vars+non_linear_vars])
X_test = scaler.transform(X_test[linear_vars+non_linear_vars])


# In[29]:


# for each linear variable I build a linear regression, a support vector machine with a linear kernel and
# a random forest, and the idea is to compare the mean squared error on the test set.

for i in range(len(linear_vars)):
    print('variable: ', linear_vars[i])
    linreg = LinearRegression()
    linreg.fit(pd.Series(X_train[:,i]).to_frame(), y_train)
    print('Test set')
    pred = linreg.predict(pd.Series(X_test[:,i]).to_frame())
    print('Linear Regression mse: {}'.format(mean_squared_error(y_test, pred)))
    print()

    rf = RandomForestRegressor(n_estimators=5, random_state=39, max_depth=2,min_samples_leaf=100)
    rf.fit(pd.Series(X_train[:,i]).to_frame(), y_train)
    print('Test set')
    pred = rf.predict(pd.Series(X_test[:,i]).to_frame())
    print('Random Forests mse: {}'.format(mean_squared_error(y_test, pred)))
    print()
    print()


# For most of the "linearly related" variables, the linear regression model is at least as good, if not better, than the random forest at estimating the Sale Price. Compare the mse of the test sets for the three different models.

# In[30]:


# for each non-linear variable I build a linear regression, a support vector machine with a linear kernel and
# a random forest, and the idea is to compare the mean squared error on the test set.

for i in [4,5]:
    print('variable: ', non_linear_vars[i-4])
    linreg = LinearRegression()
    linreg.fit(pd.Series(X_train[:,i]).to_frame(), y_train)
    print('Test set')
    pred = linreg.predict(pd.Series(X_test[:,i]).to_frame())
    print('Linear Regression mse: {}'.format(mean_squared_error(y_test, pred)))
    print()


    rf = RandomForestRegressor(n_estimators=5, random_state=39, max_depth=2,min_samples_leaf=100)
    rf.fit(pd.Series(X_train[:,i]).to_frame(), y_train)
    print('Test set')
    pred = rf.predict(pd.Series(X_test[:,i]).to_frame())
    print('Random Forests mse: {}'.format(mean_squared_error(y_test, pred)))
    print()
    print()


# Random Forests seem to be better for the first variable, and not too much of a difference for the second one.

# # Machine learning model performance when built using variables "linearly" related to the Sale Price

# In[31]:


linreg = LinearRegression()
linreg.fit(X_train[:,0:3], y_train)
print('Test set')
pred = linreg.predict(X_test[:,0:3])
print('Linear Regression mse: {}'.format(mean_squared_error(y_test, pred)))
print()

rf = RandomForestRegressor(n_estimators=5, random_state=39, max_depth=2,min_samples_leaf=100)
rf.fit(X_train[:,0:3], y_train)
print('Test set')
pred = rf.predict(X_test[:,0:3])
print('Random Forests mse: {}'.format(mean_squared_error(y_test, pred)))
print()


# Linear machine learning algorithms make betters predictions than random forests when trained on variables that show a somewhat linear relationship to the outcome, in this case, Sale Price.

# # Machine learning models performance when using variables not "linearly" related to Sale Price

# In[32]:


linreg = LinearRegression()
linreg.fit(X_train[:,4:5], y_train)
print('Test set')
pred = linreg.predict(X_test[:,4:5])
print('Linear Regression mse: {}'.format(mean_squared_error(y_test, pred)))
print()


rf = RandomForestRegressor(n_estimators=5, random_state=39, max_depth=2,min_samples_leaf=100)
rf.fit(X_train[:,4:5], y_train)
print('Test set')
pred = rf.predict(X_test[:,4:5])
print('Random Forests mse: {}'.format(mean_squared_error(y_test, pred)))
print()
print()


# However, when building a model using non-linear variables, alternative models like Random Forests may make better predictions. This is however, to be interpreted with caution, because Random Forests are good at predicting the Sale Price within the ranges of prices observed in the training dataset, but will not do a great job at inferring prices above or below the ranges on the training set.
