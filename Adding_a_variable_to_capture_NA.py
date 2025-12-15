#!/usr/bin/env python
# coding: utf-8

# # Adding a variable to capture NA

# In previous lectures we studied how to replace missing values by mean/median imputation or by extracting a random sample of the variable for those instances where data is available, and using those values to replace the missing values. We also discussed that these 2 methods assume that the missing data are missing completely at random (MCAR).
# 
# So what if the data are not missing completely at random? By using this procedure, we would be missing important, predictive information.
# 
# How can we prevent that?
# 
# We can capture the importance of missingness by creating an additional variable indicating whether the data was missing for that observation (1) or not (0). The additional variable is a binary variable: it takes only the values 0 and 1, 0 indicating that a value was present for that observation, and 1 indicating that the value was missing for that observation.

# # Advantages

# Easy to implement
# Captures the importance of missingess if there is one

# # Disadvantages

# Expands the feature space

# This method of imputation will add 1 variable per variable in the dataset with missing values. So if a dataset contains 10 features, and all of them have missing values, we will end up with a dataset with 20 features. The original features where we replaced the missing values by the mean/median (or random sampling), and additional 10 features, indicating for each of the variables, whether the value was missing or not.
# 
# This may not be a problem in datasets with tens to a few hundreds of variables, but if your original dataset contains thousands of variables, by creating an additional variable to indicate NA, you will end up with very big datasets.
# 
# In addition, data tends to be missing for the same observation on multiple variables, so it may also be the case, that many of your added variables will be actually similar to each other.

# # Final note

# Typically, mean/median imputation is done together with adding a variable to capture those observations where the data was missing (see lecture "Replacing NA with the median/mean"), thus covering 2 angles: if the data was missing completely at random, this would be contemplated by the mean imputation, and if it wasn't this would be captured by the additional variable.
# 
# In addition, both methods are extremely straight forward to implement, and therefore are a top choice in data science competitions. See for example the winning solution of the KDD 2009 cup: "Winning the KDD Cup Orange Challenge with Ensemble Selection" (http://www.mtome.com/Publications/CiML/CiML-v3-book.pdf)

# # Real Life example

# # Predicting Survival on the Titanic: understanding society behaviour and beliefs

# Perhaps one of the most infamous shipwrecks in history, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 people on board. Interestingly, by analysing the probability of survival based on few attributes like gender, age, and social status, we can make very accurate predictions on which passengers would survive. Some groups of people were more likely to survive than others, such as women, children, and the upper-class. Therefore, we can learn about the society priorities and privileges at the time.

# # Predicting Sale Price of Houses

# The problem at hand aims to predict the final sale price of homes based on different explanatory variables describing aspects of residential homes. Predicting house prices is useful to identify fruitful investments, or to determine whether the price advertised for a house is over or underestimated, before making a buying judgment.
# 
# =============================================================================
# 
# In the following cells, I will show the mean/median imputation + creating an additional variable to indicate missingness on the Titanic and House Price datasets from Kaggle.
# 
# If you haven't downloaded the dataset yet, in the lecture "Guide to setting up your computer" in section 1, you can find the details on how to do so.

# In[1]:


import pandas as pd
import numpy as np

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


# In[18]:


# load the Titanic Dataset with a few variables for demonstration

data = pd.read_csv('C:/Users/sidda/Downloads/titanic.csv', usecols = ['Age', 'Fare','Survived'])
data.head()


# In[4]:


# let's look at the percentage of NA
data.isnull().mean()


# In[5]:


# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(data, data.Survived, test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape


# In[6]:


# create variable indicating missingness

X_train['Age_NA'] = np.where(X_train['Age'].isnull(), 1, 0)
X_test['Age_NA'] = np.where(X_test['Age'].isnull(), 1, 0)

X_train.head()


# In[7]:


# we can see that mean and median are similar. So I will replace with the median
X_train.Age.mean(), X_train.Age.median(),


# In[8]:


# let's replace the NA with the median value in the training set
X_train['Age'].fillna(X_train.Age.median(), inplace=True)
X_test['Age'].fillna(X_train.Age.median(), inplace=True)

X_train.head(20)


# # Logistic Regression

# In[9]:


scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))

X_train.columns = ['Survived','Age','Fare','Age_NA']
X_test.columns = ['Survived','Age','Fare','Age_NA']


# In[12]:


from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

# --- Step 1: Impute missing values with median ---
imputer = SimpleImputer(strategy='median')

X_train_imp = X_train.copy()
X_test_imp = X_test.copy()

# Impute Age and Fare
X_train_imp[['Age','Fare']] = imputer.fit_transform(X_train[['Age','Fare']])
X_test_imp[['Age','Fare']] = imputer.transform(X_test[['Age','Fare']])

# If using Age_NA indicator, fill Age missing values and keep indicator
if 'Age_NA' in X_train_imp.columns:
    # Age_NA should be 1 if Age was originally missing
    X_train_imp['Age_NA'] = X_train['Age'].isna().astype(int)
    X_test_imp['Age_NA'] = X_test['Age'].isna().astype(int)

# --- Step 2: Logistic Regression without Age_NA ---
logit = LogisticRegression(random_state=44, C=1000)
logit.fit(X_train_imp[['Age','Fare']], y_train)

print('Train set (Age + Fare)')
pred_train = logit.predict_proba(X_train_imp[['Age','Fare']])[:,1]
print('ROC-AUC:', roc_auc_score(y_train, pred_train))

pred_test = logit.predict_proba(X_test_imp[['Age','Fare']])[:,1]
print('Test set ROC-AUC:', roc_auc_score(y_test, pred_test))

# --- Step 3: Logistic Regression with Age_NA indicator ---
if 'Age_NA' in X_train_imp.columns:
    logit2 = LogisticRegression(random_state=44, C=1000)
    logit2.fit(X_train_imp[['Age','Age_NA','Fare']], y_train)

    print('\nTrain set (Age + Age_NA + Fare)')
    pred_train2 = logit2.predict_proba(X_train_imp[['Age','Age_NA','Fare']])[:,1]
    print('ROC-AUC:', roc_auc_score(y_train, pred_train2))

    pred_test2 = logit2.predict_proba(X_test_imp[['Age','Age_NA','Fare']])[:,1]
    print('Test set ROC-AUC:', roc_auc_score(y_test, pred_test2))


# # Support Vector Machine

# In[14]:


from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import pandas as pd

# --- Step 1: Impute missing values ---
imputer = SimpleImputer(strategy='median')

X_train_imp = X_train.copy()
X_test_imp = X_test.copy()

# Impute Age and Fare
X_train_imp[['Age','Fare']] = imputer.fit_transform(X_train[['Age','Fare']])
X_test_imp[['Age','Fare']] = imputer.transform(X_test[['Age','Fare']])

# Add Age_NA indicator if needed
if 'Age_NA' in X_train.columns:
    X_train_imp['Age_NA'] = X_train['Age'].isna().astype(int)
    X_test_imp['Age_NA'] = X_test['Age'].isna().astype(int)

# --- Step 2: SVC without Age_NA ---
SVM_model = SVC(random_state=44, probability=True, max_iter=-1, kernel='linear')
SVM_model.fit(X_train_imp[['Age','Fare']], y_train)

print('Train set (Age + Fare)')
pred_train = SVM_model.predict_proba(X_train_imp[['Age','Fare']])[:,1]
print('SVC ROC-AUC:', roc_auc_score(y_train, pred_train))

pred_test = SVM_model.predict_proba(X_test_imp[['Age','Fare']])[:,1]
print('Test set ROC-AUC:', roc_auc_score(y_test, pred_test))

# --- Step 3: SVC with Age_NA indicator ---
if 'Age_NA' in X_train_imp.columns:
    SVM_model2 = SVC(random_state=44, probability=True, max_iter=-1, kernel='linear')
    SVM_model2.fit(X_train_imp[['Age','Age_NA','Fare']], y_train)

    print('\nTrain set (Age + Age_NA + Fare)')
    pred_train2 = SVM_model2.predict_proba(X_train_imp[['Age','Age_NA','Fare']])[:,1]
    print('SVC ROC-AUC:', roc_auc_score(y_train, pred_train2))

    pred_test2 = SVM_model2.predict_proba(X_test_imp[['Age','Age_NA','Fare']])[:,1]
    print('Test set ROC-AUC:', roc_auc_score(y_test, pred_test2))




# In the titanic dataset, including a variable to indicate missingness for Age did not show an improvement in the performance of the logistic regression and barely the support vector machine.

# # House Sale Dataset

# In[27]:


# we are going to train a model on the following variables,

cols_to_use = ['id', 'date', 'price', 'bedrooms','bathrooms', 'floors',
               'view', 'grade', 'zipcode']


# In[29]:


import pandas as pd

# Load the CSV without restricting columns
df = pd.read_csv('C:/Users/sidda/Downloads/kc_house_data.csv')

# See all column names
print("Columns in CSV:", df.columns.tolist())

# Define only the columns that exist in your dataset
existing_cols = [c for c in cols_to_use + ['SalePrice'] if c in df.columns]

# Load only the existing columns
data = df[existing_cols]
print("Data shape:", data.shape)
data.head()


# In[30]:


# let's inspect the columns  with missing values
data.isnull().mean()


# In[32]:


# let's separate into training and testing set
from sklearn.model_selection import train_test_split

target_col = 'price'  # replace with the actual name from data.columns
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=[target_col]), 
    data[target_col], 
    test_size=0.3, 
    random_state=0
)

print(X_train.shape, X_test.shape)


# We observed that the numerical variables are not normally distributed. In particular, most of them apart from YearBuilt are skewed.

# In[39]:


# let's make a function to replace the NA with median or 0s

def impute_na(df, variable, median):
    df[variable+'_NA'] = np.where(df[variable].isnull(), 1, 0)
    df[variable].fillna(median, inplace=True)
    


# In[44]:


# let's look at the median of the variables with NA

cols_to_check = ['sqft_basement', 'yr_built', 'yr_renovated']
existing_cols = [col for col in cols_to_check if col in X_train.columns]
X_train[existing_cols].median()



# In[46]:


# let's impute the NA with  the median
# remember that we need to impute with the median for the train set, and then propagate to test set

median = X_train.sqft_basement.median()
impute_na(X_train, 'sqft_basement', median)
impute_na(X_test, 'sqft_basement', median)


# In[47]:


median = X_train.yr_built.median()
impute_na(X_train, 'yr_built', median)
impute_na(X_test, 'yr_built', median)


# In[48]:


median = X_train.yr_renovated.median()
impute_na(X_train, 'yr_renovated', median)
impute_na(X_test, 'yr_renovated', median)


# In[49]:


X_train.isnull().mean()


# In[50]:


# create a list with the untransformed columns
cols_to_use_no_na = X_train.columns[:-4]
cols_to_use_no_na


# In[53]:


cols_to_use = list(X_train.columns)
cols_to_use.remove('yr_built')
cols_to_use


# In[57]:


# let's standarise the dataset
from sklearn.preprocessing import StandardScaler

# Select only numeric columns for scaling
numeric_cols_no_na = [col for col in cols_to_use_no_na if np.issubdtype(X_train[col].dtype, np.number)]
numeric_cols_all = [col for col in cols_to_use_all if np.issubdtype(X_train[col].dtype, np.number)]

print("Numeric columns without NA:", numeric_cols_no_na)
print("Numeric columns (all):", numeric_cols_all)

# Standardize numeric columns without NA
scaler_no_na = StandardScaler()
X_train_no_na_scaled = scaler_no_na.fit_transform(X_train[numeric_cols_no_na])
X_test_no_na_scaled = scaler_no_na.transform(X_test[numeric_cols_no_na])

# Standardize numeric columns including NA-handled ones
scaler_all = StandardScaler()
X_train_all_scaled = scaler_all.fit_transform(X_train[numeric_cols_all])
X_test_all_scaled = scaler_all.transform(X_test[numeric_cols_all])

print("Standardization complete.")


# # Linear Regression

# In[59]:


# we compare the models built using Age filled with median, vs Age filled with median + additional
# variable indicating missingness

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Linear Regression using only numeric columns without NA
linreg = LinearRegression()
linreg.fit(X_train_no_na_scaled, y_train)
print('Train set')
pred = linreg.predict(X_train_no_na_scaled)
print('Linear Regression MSE:', mean_squared_error(y_train, pred))
print('Test set')
pred = linreg.predict(X_test_no_na_scaled)
print('Linear Regression MSE:', mean_squared_error(y_test, pred))
print()

# Linear Regression using all numeric columns (including NA-handled features)
linreg = LinearRegression()
linreg.fit(X_train_all_scaled, y_train)
print('Train set')
pred = linreg.predict(X_train_all_scaled)
print('Linear Regression MSE:', mean_squared_error(y_train, pred))
print('Test set')
pred = linreg.predict(X_test_all_scaled)
print('Linear Regression MSE:', mean_squared_error(y_test, pred))


# In[60]:


#  what is the difference in price estimated by the 2 models?

2212393764-2197999822


# Here, when we build a model using the additional variable to capture missingness of data, we observe in the test set that the mse is smaller. This means that the difference between the real value and the estimated value is smaller, and thus our model performs better.
# 
# There is a difference of ~14 million between the model that replaces with the median and the one that uses median imputation in combination with the additional variables to capture missingness. So even when the difference in mse seems small, when we boil it down to business value, the impact is massive.
# 
# For a discussion on why the median imputation is not enough in this dataset, refer to lecture "Replacing NA by mean or median"

# In[62]:


# we compare the models built using Age filled with median, vs Age filled with median + additional
# variable indicating missingness

#  Ridge, is a regularised linear regression.

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Ridge Regression using numeric columns without NA
ridge = Ridge(random_state=30, max_iter=5, tol=100, alpha=10)
ridge.fit(X_train_no_na_scaled, y_train)
print('Train set')
pred = ridge.predict(X_train_no_na_scaled)
print('Ridge Regression MSE:', mean_squared_error(y_train, pred))
print('Test set')
pred = ridge.predict(X_test_no_na_scaled)
print('Ridge Regression MSE:', mean_squared_error(y_test, pred))
print()

# Ridge Regression using all numeric columns (including NA-handled features)
ridge = Ridge(random_state=30, max_iter=5, tol=100, alpha=10)
ridge.fit(X_train_all_scaled, y_train)
print('Train set')
pred = ridge.predict(X_train_all_scaled)
print('Ridge Regression MSE:', mean_squared_error(y_train, pred))
print('Test set')
pred = ridge.predict(X_test_all_scaled)
print('Ridge Regression MSE:', mean_squared_error(y_test, pred))


# We observe the same conclusion when using regularised linear regression.
