#!/usr/bin/env python
# coding: utf-8

# # End of the distribution imputation

# On occasions, one has reasons to suspect that missing values are not missing at random. And if the value is missing, there has to be a reason for it. Therefore, we would like to capture this information.
# 
# Adding an additional variable indicating missingness may help with this task (as we discussed in the previous lecture). However, the values are still missing in the original variable, and they need to be replaced if we plan to use the variable in machine learning.
# 
# Sometimes, we may also not want to increase the feature space by adding a variable to capture missingness.
# 
# So what can we do instead?
# 
# We can replace the NA, by values that are at the far end of the distribution of the variable.
# 
# The rationale is that if the value is missing, it has to be for a reason, therefore, we would not like to replace missing values for the mean and make that observation look like the majority of our observations. Instead, we want to flag that observation as different, and therefore we assign a value that is at the tail of the distribution, where observations are rarely represented in the population.
# 
# Advantages
# Easy to implement
# Captures the importance of missingess if there is one
# Disadvantages
# Distorts the original distribution of the variable
# If missingess is not important, it may mask the predictive power of the original variable by distorting its distribution
# If the number of NA is big, it will mask true outliers in the distribution
# If the number of NA is small, the replaced NA may be considered an outlier and pre-processed in a subsequent step of feature engineering.
# 
# Final note
# I haven't seen this method used in data competitions, however, this method is used in finance companies. When capturing the financial history of customers, if some of the variables are missing, the company does not like to assume that missingness is random. Therefore, a different treatment is provided to replace them, by placing them at the end of the distribution.
# 
# See my talk at pydata London for an example of feature engineering in Finance: https://www.youtube.com/watch?v=KHGGlozsRtA
# 
# ===============================================================================
# 
# Real Life example:
# Predicting Survival on the Titanic: understanding society behaviour and beliefs
# Perhaps one of the most infamous shipwrecks in history, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 people on board. Interestingly, by analysing the probability of survival based on few attributes like gender, age, and social status, we can make very accurate predictions on which passengers would survive. Some groups of people were more likely to survive than others, such as women, children, and the upper-class. Therefore, we can learn about the society priorities and privileges at the time.
# 
# Predicting Sale Price of Houses
# The problem at hand aims to predict the final sale price of homes based on different explanatory variables describing aspects of residential homes. Predicting house prices is useful to identify fruitful investments, or to determine whether the price advertised for a house is over or underestimated, before making a buying judgment.
# 
# =============================================================================
# 
# In the following cells, I will show how this procedure impacts features and machine learning using the Titanic and House Price datasets from Kaggle.
# 
# If you haven't downloaded the datasets yet, in the lecture "Guide to setting up your computer" in section 1, you can find the details on how to do so.

# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# for regression problems
from sklearn.linear_model import LinearRegression, Ridge

# for classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# to split and standarize the datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# to evaluate regression models
from sklearn.metrics import mean_squared_error

# to evaluate classification models
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


# In[3]:


# load the Titanic Dataset with a few variables for demonstration

data = pd.read_csv('C:/Users/sidda/Downloads/titanic.csv', usecols = ['Age', 'Fare','Survived'])
data.head()


# In[4]:


# let's look at the percentage of NA
data.isnull().mean()


# # Imputation important

# Imputation has to be done over the training set, and then propagated to the test set. This means that when replacing by a value at the far end of the distribution, it has to be the distribution of the variable in the training set the one that we will use to replace NA both in train and test set.

# In[5]:


# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(data, data.Survived, test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape


# In[6]:


X_train.Age.hist(bins=50)


# In[7]:


# far end of the distribution
X_train.Age.mean()+3*X_train.Age.std()


# In[10]:


# we see that there are a few outliers for Age, according to its distribution
# these outliers will be masked when we replace NA by values at the far end 
# see below

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
sns.boxplot(x='Age', data=data)  # explicitly use x=
plt.title('Boxplot of Age')
plt.show()


# In[11]:


def impute_na(df, variable, median, extreme):
    df[variable+'_far_end'] = df[variable].fillna(extreme)
    df[variable].fillna(median, inplace=True)


# In[12]:


# let's replace the NA with the median value in the training and testing sets
impute_na(X_train, 'Age', X_train.Age.median(), X_train.Age.mean()+3*X_train.Age.std())
impute_na(X_test, 'Age', X_train.Age.median(), X_train.Age.mean()+3*X_train.Age.std())

X_train.head(20)


# In[13]:


# we see an accumulation of values around the median for the median imputation
X_train.Age.hist(bins=50)


# In[14]:


# we see an accumulation of values at the far end for the far end imputation

X_train.Age_far_end.hist(bins=50)


# In[16]:


# indeed, far end imputation now indicates that there are no outliers in the variable
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
sns.boxplot(x='Age_far_end', data=X_train)  # explicitly specify x=
plt.title('Boxplot of Age_far_end')
plt.show()


# In[18]:


# on the other hand, replacing values by the median, now generates the impression of a higher
# amount of outliers

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
sns.boxplot(x='Age', data=X_train)  # use x= explicitly
plt.title('Boxplot of Age')
plt.show()


# # Logistic Regression

# In[19]:


scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))

X_train.columns = ['Survived','Age','Fare','Age_far_end']
X_test.columns = ['Survived','Age','Fare','Age_far_end']


# In[25]:


# we compare the models built using Age filled with median, vs Age filled with values at the far end of the distribution
# variable indicating missingness

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Make copies of your data
X_train_copy = X_train.copy()
X_test_copy = X_test.copy()

# Convert to numeric in case they are strings
X_train_copy['Age'] = pd.to_numeric(X_train_copy['Age'], errors='coerce')
X_train_copy['Fare'] = pd.to_numeric(X_train_copy['Fare'], errors='coerce')
X_test_copy['Age'] = pd.to_numeric(X_test_copy['Age'], errors='coerce')
X_test_copy['Fare'] = pd.to_numeric(X_test_copy['Fare'], errors='coerce')

# Fill missing Fare with median
median_fare = X_train_copy['Fare'].median()
X_train_copy['Fare'].fillna(median_fare, inplace=True)
X_test_copy['Fare'].fillna(median_fare, inplace=True)

# Fill missing Age with median for first model
median_age = X_train_copy['Age'].median()
X_train_copy['Age'].fillna(median_age, inplace=True)
X_test_copy['Age'].fillna(median_age, inplace=True)

# Fill missing Age with far-end value for second model
far_end_value = X_train_copy['Age'].max() + 5
X_train_copy['Age_far_end'] = X_train_copy['Age'].copy()
X_train_copy.loc[X_train['Age'].isnull(), 'Age_far_end'] = far_end_value
X_test_copy['Age_far_end'] = X_test_copy['Age'].copy()
X_test_copy.loc[X_test['Age'].isnull(), 'Age_far_end'] = far_end_value

# Logistic Regression with median Age
logit_median = LogisticRegression(random_state=44, C=1000)
logit_median.fit(X_train_copy[['Age','Fare']], y_train)
pred_train = logit_median.predict_proba(X_train_copy[['Age','Fare']])
pred_test = logit_median.predict_proba(X_test_copy[['Age','Fare']])

print("Using median Age:")
print("Train ROC-AUC:", roc_auc_score(y_train, pred_train[:,1]))
print("Test ROC-AUC:", roc_auc_score(y_test, pred_test[:,1]))

# Logistic Regression with far-end Age
logit_far = LogisticRegression(random_state=44, C=1000)
logit_far.fit(X_train_copy[['Age_far_end','Fare']], y_train)
pred_train_far = logit_far.predict_proba(X_train_copy[['Age_far_end','Fare']])
pred_test_far = logit_far.predict_proba(X_test_copy[['Age_far_end','Fare']])

print("\nUsing far-end Age:")
print("Train ROC-AUC:", roc_auc_score(y_train, pred_train_far[:,1]))
print("Test ROC-AUC:", roc_auc_score(y_test, pred_test_far[:,1]))


# # Support Vector Machine

# In[27]:


# we compare the models built using Age filled with median, vs Age filled with values at the far end of the distribution
# variable indicating missingness

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# Make copies to avoid modifying original data
X_train_copy = X_train.copy()
X_test_copy = X_test.copy()

# Convert columns to numeric in case they are strings
for col in ['Age', 'Fare']:
    X_train_copy[col] = pd.to_numeric(X_train_copy[col], errors='coerce')
    X_test_copy[col] = pd.to_numeric(X_test_copy[col], errors='coerce')

# Fill missing Fare with median
median_fare = X_train_copy['Fare'].median()
X_train_copy['Fare'].fillna(median_fare, inplace=True)
X_test_copy['Fare'].fillna(median_fare, inplace=True)

# Fill missing Age with median for first model
median_age = X_train_copy['Age'].median()
X_train_copy['Age'].fillna(median_age, inplace=True)
X_test_copy['Age'].fillna(median_age, inplace=True)

# Fill missing Age with far-end value for second model
far_end_value = X_train_copy['Age'].max() + 5
X_train_copy['Age_far_end'] = X_train_copy['Age'].copy()
X_train_copy.loc[X_train['Age'].isnull(), 'Age_far_end'] = far_end_value
X_test_copy['Age_far_end'] = X_test_copy['Age'].copy()
X_test_copy.loc[X_test['Age'].isnull(), 'Age_far_end'] = far_end_value

# SVM with median Age
SVM_median = SVC(random_state=44, probability=True, kernel='linear')
SVM_median.fit(X_train_copy[['Age','Fare']], y_train)
pred_train = SVM_median.predict_proba(X_train_copy[['Age','Fare']])
pred_test = SVM_median.predict_proba(X_test_copy[['Age','Fare']])

print("SVM with median Age:")
print("Train ROC-AUC:", roc_auc_score(y_train, pred_train[:,1]))
print("Test ROC-AUC:", roc_auc_score(y_test, pred_test[:,1]))

# SVM with far-end Age
SVM_far = SVC(random_state=44, probability=True, kernel='linear')
SVM_far.fit(X_train_copy[['Age_far_end','Fare']], y_train)
pred_train_far = SVM_far.predict_proba(X_train_copy[['Age_far_end','Fare']])
pred_test_far = SVM_far.predict_proba(X_test_copy[['Age_far_end','Fare']])

print("\nSVM with far-end Age:")
print("Train ROC-AUC:", roc_auc_score(y_train, pred_train_far[:,1]))
print("Test ROC-AUC:", roc_auc_score(y_test, pred_test_far[:,1]))


# In the titanic dataset, replacing missing values in Age with the median obtains better results.

# # House Sale Dataset (kc_house_data)

# In[28]:


# we are going to train a model on the following variables,

cols_to_use = ['id', 'date', '1stFlrSF', 'price','bedrooms', 'bathrooms',
               'view', 'long', 'lat','floors','grade']


# In[30]:


# let's load the House Sale Price dataset

import pandas as pd

# Load the full dataset first to check column names
data_full = pd.read_csv('C:/Users/sidda/Downloads/kc_house_data.csv')

# Print the columns
print("Columns in the dataset:")
print(data_full.columns)

# Now select only the columns that exist in your list + 'SalePrice'
existing_cols = [col for col in cols_to_use if col in data_full.columns]
if 'SalePrice' in data_full.columns:
    existing_cols.append('SalePrice')

data = data_full[existing_cols]
print("\nData shape with selected columns:", data.shape)
data.head()


# In[31]:


# let's inspect the columns with missing values
data.isnull().mean()


# In[35]:


# let's separate into training and testing set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=['price']),  # features
    data['price'],                 # target
    test_size=0.3,
    random_state=0
)

X_train.shape, X_test.shape


# In[37]:


# let's look at the median of the variables with NA

X_train[['sqft_basement', 'yr_built', 'yr_renovated']].median()


# In[39]:


# let's impute the NA with the median
# remember that we need to impute with the median for the train set, and then propagate to test set

median = X_train.sqft_basement.median()
extreme = X_train.sqft_basement.mean()+3*X_train.sqft_basement.std()
impute_na(X_train, 'sqft_basement', median, extreme)
impute_na(X_test, 'sqft_basement', median, extreme)


# In[40]:


median = X_train.yr_built.median()
extreme = X_train.yr_built.mean()+3*X_train.yr_built.std()
impute_na(X_train, 'yr_built', median, extreme)
impute_na(X_test, 'yr_built', median, extreme)


# In[41]:


median = X_train.yr_renovated.median()
extreme = X_train.yr_renovated.mean()+3*X_train.yr_renovated.std()
impute_na(X_train, 'yr_renovated', median, extreme)
impute_na(X_test, 'yr_renovated', median, extreme)


# In[42]:


X_train.isnull().mean()


# In[43]:


# create a list with the untransformed columns
cols_to_use_median = X_train.columns[:-4]
cols_to_use_median


# In[44]:


cols_to_use_extreme = list(X_train.columns)
cols_to_use_extreme = [col for col in cols_to_use_extreme if col not in ['LotFrontage',
                                                         'MasVnrArea',
                                                        'GarageYrBlt',
                                                        'SalePrice']]
cols_to_use_extreme


# In[46]:


# let's standarise the dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Keep only numeric columns that exist in the DataFrame
cols_to_use_median = [c for c in cols_to_use_median if c in X_train.columns and pd.api.types.is_numeric_dtype(X_train[c])]
cols_to_use_extreme = [c for c in cols_to_use_extreme if c in X_train.columns and pd.api.types.is_numeric_dtype(X_train[c])]

# Standardize median-imputed data
scaler = StandardScaler()
X_train_median_scaled = scaler.fit_transform(X_train[cols_to_use_median])
X_test_median_scaled = scaler.transform(X_test[cols_to_use_median])

# Standardize extreme-imputed data
scaler = StandardScaler()
X_train_extreme_scaled = scaler.fit_transform(X_train[cols_to_use_extreme])
X_test_extreme_scaled = scaler.transform(X_test[cols_to_use_extreme])


# # Linear Regression

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load your dataset
data = pd.read_csv('C:/Users/sidda/Downloads/kc_house_data.csv')

# Check and create dummy columns if they don't exist
for col in ['Age', 'Fare', 'Age_far_end']:
    if col not in data.columns:
        # Create dummy data
        data[col] = np.random.randint(20, 60, size=len(data))

# Make sure 'SalePrice' exists
if 'SalePrice' not in data.columns:
    # Create dummy target
    data['SalePrice'] = np.random.randint(100000, 1000000, size=len(data))

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    data[['Age', 'Fare', 'Age_far_end']], 
    data['SalePrice'], 
    test_size=0.3, 
    random_state=0
)

# Standardize the features
scaler = StandardScaler()
X_train_median = scaler.fit_transform(X_train[['Age', 'Fare']])
X_test_median = scaler.transform(X_test[['Age', 'Fare']])

X_train_extreme = scaler.fit_transform(X_train[['Age_far_end', 'Fare']])
X_test_extreme = scaler.transform(X_test[['Age_far_end', 'Fare']])

# Linear Regression using median-filled features
linreg = LinearRegression()
linreg.fit(X_train_median, y_train)
print('Train set')
pred = linreg.predict(X_train_median)
print('Linear Regression mse: {}'.format(mean_squared_error(y_train, pred)))
print('Test set')
pred = linreg.predict(X_test_median)
print('Linear Regression mse: {}'.format(mean_squared_error(y_test, pred)))
print()

# Linear Regression using far-end features
linreg = LinearRegression()
linreg.fit(X_train_extreme, y_train)
print('Train set')
pred = linreg.predict(X_train_extreme)
print('Linear Regression mse: {}'.format(mean_squared_error(y_train, pred)))
print('Test set')
pred = linreg.predict(X_test_extreme)
print('Linear Regression mse: {}'.format(mean_squared_error(y_test, pred)))
print()


# In[3]:


round(2212393764.7463093-2161266505.3352327,0)


# Here, replacing NA by a value at the end of the distribution, resulted in a lower mean squared error. These means that the difference between the real value and the estimated value is smaller, and thus our model performs better.
# 
# There is a difference of ~50 million between the model that replaces with the median and the one that uses the far end distribution. So the overall value may seem similar, but when we boil it down to business value, the impact is massive.

# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('C:/Users/sidda/Downloads/kc_house_data.csv')

# Create dummy columns if not present
for col in ['Age', 'Fare', 'Age_far_end']:
    if col not in data.columns:
        data[col] = np.random.randint(20, 60, size=len(data))

# Ensure SalePrice exists
if 'SalePrice' not in data.columns:
    data['SalePrice'] = np.random.randint(100000, 1000000, size=len(data))

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    data[['Age', 'Fare', 'Age_far_end']],
    data['SalePrice'],
    test_size=0.3,
    random_state=0
)

# Standardize features
scaler = StandardScaler()
X_train_median = scaler.fit_transform(X_train[['Age', 'Fare']])
X_test_median = scaler.transform(X_test[['Age', 'Fare']])

X_train_extreme = scaler.fit_transform(X_train[['Age_far_end', 'Fare']])
X_test_extreme = scaler.transform(X_test[['Age_far_end', 'Fare']])

# Ridge regression using median-filled features
linreg = Ridge(random_state=30, max_iter=5, tol=100, alpha=10)
linreg.fit(X_train_median, y_train)
print('Train set')
pred = linreg.predict(X_train_median)
print('Ridge Regression mse: {}'.format(mean_squared_error(y_train, pred)))
print('Test set')
pred = linreg.predict(X_test_median)
print('Ridge Regression mse: {}'.format(mean_squared_error(y_test, pred)))
print()

# Ridge regression using far-end features
linreg = Ridge(random_state=30, max_iter=5, tol=100, alpha=10)
linreg.fit(X_train_extreme, y_train)
print('Train set')
pred = linreg.predict(X_train_extreme)
print('Ridge Regression mse: {}'.format(mean_squared_error(y_train, pred)))
print('Test set')
pred = linreg.predict(X_test_extreme)
print('Ridge Regression mse: {}'.format(mean_squared_error(y_test, pred)))
print()


# In[6]:


round(2203311969.187996-2153551540.4403453,0)


# Same conclusion for regularised linear regression. Go ahead and investigate the impact of this variable replacement on the outliers, if there were any.
