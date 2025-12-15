#!/usr/bin/env python
# coding: utf-8

# # Arbitrary value imputation

# Replacing the NA by artitrary values should be used when there are reasons to believe that the NA are not missing at random. In situations like this, we would not like to replace with the median or the mean, and therefore make the NA look like the majority of our observations.
# 
# Instead, we want to flag them. We want to capture the missingness somehow.
# 
# In previous lectures we saw 2 methods to do this:
# 
# adding an additional binary variable to indicate whether the value is missing (1) or not (0)
# 
# replacing the NA by a value at a far end of the distribution
# 
# Here, I suggest an alternative to option 2, which I have seen in several Kaggle competitions. It consists of replacing the NA by an arbitrary value. Any of your creation, but ideally different from the median/mean/mode, and not within the normal values of the variable.
# 
# The problem consists in deciding which arbitrary value to choose.
# 
# Advantages
# Easy to implement
# Captures the importance of missingess if there is one
# Disadvantages
# Distorts the original distribution of the variable
# If missingess is not important, it may mask the predictive power of the original variable by distorting its distribution
# Hard to decide which value to use If the value is outside the distribution it may mask or create outliers
# Final note
# When variables are captured by third parties, like credit agencies, they place arbitrary numbers already to signal this missingness. So if not common practice in data competitions, it is common practice in real life data collections.
# 
# ===============================================================================
# 
# Real Life example:
# Predicting Survival on the Titanic: understanding society behaviour and beliefs
# Perhaps one of the most infamous shipwrecks in history, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 people on board. Interestingly, by analysing the probability of survival based on few attributes like gender, age, and social status, we can make very accurate predictions on which passengers would survive. Some groups of people were more likely to survive than others, such as women, children, and the upper-class. Therefore, we can learn about the society priorities and privileges at the time.
# 
# =============================================================================

# In[1]:


import pandas as pd
import numpy as np

# for classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# to split and standarize the datasets
from sklearn.model_selection import train_test_split

# to evaluate classification models
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# load the Titanic Dataset with a few variables for demonstration

data = pd.read_csv('C:/Users/sidda/Downloads/titanic.csv', usecols = ['Age', 'Fare','Survived'])
data.head()


# In[3]:


# let's look at the percentage of NA
data.isnull().mean()


# In[4]:


# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(data, data.Survived, test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape


# In[5]:


def impute_na(df, variable):
    df[variable+'_zero'] = df[variable].fillna(0)
    df[variable+'_hundred']= df[variable].fillna(100)
    


# In[6]:


# let's replace the NA with the median value in the training set
impute_na(X_train, 'Age')
impute_na(X_test, 'Age')

X_train.head(20)


# # Logistic Regression

# In[14]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# =========================
# 1. CREATE Age imputations
# =========================

# Age filled with 0
X_train['Age_zero'] = X_train['Age'].fillna(0)
X_test['Age_zero'] = X_test['Age'].fillna(0)

# Age filled with 100
X_train['Age_hundred'] = X_train['Age'].fillna(100)
X_test['Age_hundred'] = X_test['Age'].fillna(100)

# =========================
# 2. FIX Fare NaNs (IMPORTANT)
# =========================

fare_median = X_train['Fare'].median()
X_train['Fare'] = X_train['Fare'].fillna(fare_median)
X_test['Fare'] = X_test['Fare'].fillna(fare_median)

# =========================
# 3. LOGISTIC REGRESSION
# =========================

# ---- Age filled with ZERO ----
logit = LogisticRegression(random_state=44, C=1000)
logit.fit(X_train[['Age_zero','Fare']], y_train)

print('Train set')
pred = logit.predict_proba(X_train[['Age_zero','Fare']])
print('Logistic Regression roc-auc:', roc_auc_score(y_train, pred[:, 1]))

print('Test set')
pred = logit.predict_proba(X_test[['Age_zero','Fare']])
print('Logistic Regression roc-auc:', roc_auc_score(y_test, pred[:, 1]))

print()

# ---- Age filled with 100 ----
logit = LogisticRegression(random_state=44, C=1000)
logit.fit(X_train[['Age_hundred','Fare']], y_train)

print('Train set')
pred = logit.predict_proba(X_train[['Age_hundred','Fare']])
print('Logistic Regression roc-auc:', roc_auc_score(y_train, pred[:, 1]))

print('Test set')
pred = logit.predict_proba(X_test[['Age_hundred','Fare']])
print('Logistic Regression roc-auc:', roc_auc_score(y_test, pred[:, 1]))


# In[15]:


rf = RandomForestClassifier(n_estimators=100, random_state=39, max_depth=3)
rf.fit(X_train[['Age_zero', 'Fare']], y_train)
print('Train set zero imputation')
pred = rf.predict_proba(X_train[['Age_zero', 'Fare']])
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set zero imputation')
pred = rf.predict_proba(X_test[['Age_zero', 'Fare']])
print('Random Forests zero imputation roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
print()
rf = RandomForestClassifier(n_estimators=100, random_state=39, max_depth=3)
rf.fit(X_train[['Age_hundred', 'Fare']], y_train)
print('Train set median imputation')
pred = rf.predict_proba(X_train[['Age_hundred', 'Fare']])
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set median imputation')
pred = rf.predict_proba(X_test[['Age_hundred', 'Fare']])
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
print()


# We can see that replacing NA with 100 makes the models perform better than replacing NA with 0. This is, if you remember from the lecture "Replacing NA by mean or median" because children were more likely to survive than adults. Then filling NA with zeroes, distorts this relation and makes the models loose predictive power. See below for a re-cap.

# In[16]:


print('Average real survival of children: ', X_train[X_train.Age<15].Survived.mean())
print('Average survival of children when using Age imputed with zeroes: ', X_train[X_train.Age_zero<15].Survived.mean())
print('Average survival of children when using Age imputed with median: ', X_train[X_train.Age_hundred<15].Survived.mean())

Final notes
The arbitrary value has to be determined for each variable specifically. For example, for this dataset, the choice of replacing NA in age by 0 or 100 are valid, because none of those values are frequent in the original distribution of the variable, and they lie at the tails of the distribution.

However, if we were to replace NA in fare, those values are not good any more, because we can see that fare can take values of up to 500. So we might want to consider using 500 or 1000 to replace NA instead of 100.

As you can see this is totally arbitrary. And yet, it is used in the industry.

Typical values chose by companies are -9999 or 9999, or similar.