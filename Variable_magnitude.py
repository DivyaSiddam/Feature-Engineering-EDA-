#!/usr/bin/env python
# coding: utf-8

# # Variable magnitude

# # Does the magnitude of the variable matter?

# In Linear Regression models, the scale of variables used to estimate the output matters. Linear models are of the type y = w x + b, where the regression coefficient w represents the expected change in y for a one unit change in x (the predictor). Thus, the magnitude of w is partly determined by the magnitude of the units being used for x. If x is a distance variable, just changing the scale from kilometers to miles will cause a change in the magnitude of the coefficient.
# 
# In addition, in situations where we estimate the outcome y by contemplating multiple predictors x1, x2, ...xn, predictors with greater numeric ranges dominate over those with smaller numeric ranges.
# 
# Gradient descent converges faster when all the predictors (x1 to xn) are within a similar scale, therefore making feature scaling useful for Neural Networks as well as Logistic Regression.
# 
# In Support Vector Machines, feature scaling can decrease the time to find the support vectors.
# 
# Finally, methods using Euclidean distances or distances in general are also affected by the magnitude of the features, as Euclidean distance is sensitive to variations in the magnitude or scales of the predictors. Therefore feature scaling is required for methods that utilise distance calculations like k-nearest neighbours (KNN) and k-means clustering.
# 
# For more details on the above, follow the links in the Bonus Lecture of this section.
# 
# In summary:
# 
# Maginutd matters because:
# The regression coefficient is directly influenced by the scale of the variable
# Variables with bigger magnitudes / value range dominate over the ones with smaller magnitudes / value range
# Gradient descent converges faster when features are on similar scales
# Feature scaling helps decrease the time to find support vectors for SVMs
# Euclidean distances are sensitive to feature magnitude.
# The machine learning models affected by the magnitude of the feature are:
# Linear and Logistic Regression
# Neural Networks
# Support Vector Machines
# KNN
# K-means clustering
# Linear Discriminant Analysis (LDA)
# Principal Component Analysis (PCA)
# 
# Machine learning models insensitive to feature magnitude are the ones based on Trees:
# Classification and Regression Trees
# Random Forests
# Gradient Boosted Trees
# 
# Real Life example:
# Predicting Survival on the Titanic: understanding society behaviour and beliefs
# Perhaps one of the most infamous shipwrecks in history, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 people on board. Interestingly, by analysing the probability of survival based on few attributes like gender, age, and social status, we can make very accurate predictions on which passengers would survive. Some groups of people were more likely to survive than others, such as women, children, and the upper-class. Therefore, we can learn about the society priorities and privileges at the time.
# 
# ====================================================================================================
# 
# To download the Titanic data, go ahead to this website: https://www.kaggle.com/c/titanic/data
# 
# Click on the link 'train.csv', and then click the 'download' blue button towards the right of the screen, to download the dataset. Save it in a folder of your choice.
# 
# Note that you need to be logged in to Kaggle in order to download the datasets.
# 
# If you save it in the same directory from which you are running this notebook, and you rename the file to 'titanic.csv' then you can load it the same way I will load it below.
# 
# ====================================================================================================
# 
# In this notebook, I will demonstrate the effect of feature magnitude on the performance of different machine learning algorithms.

# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# # Load data with numerical variables only

# In[4]:


# load the numerical variables of the Titanic Dataset
data = pd.read_csv('C:/Users/sidda/Downloads/titanic.csv', usecols = ['Pclass', 'Age', 'Fare', 'Survived'])
data.head()


# In[5]:


# let's have a look at the values of those variables to get an idea of the magnitudes

data.describe()


# In[6]:


# let's now calculate the range

for col in ['Pclass', 'Age', 'Fare']:
    print(col, '_range: ', data[col].max()-data[col].min())


# The magnitude of the values of the 3 different variables and their ranges of values are quite different. Therefore, feature scaling could benefit the performance of several machine learning algorithms.

# In[7]:


# let's separate into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    data[['Pclass', 'Age', 'Fare']].fillna(0),
    data.Survived,
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape


# # Feature Scaling
# 

# For this demonstration, I will scale the features between 0 and 1. To learn more about this scaling visit the scikit-learn website: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
# 
# Briefly, the transformation is given by:
# 
# X_std = (X - X.min() / (X.max - X.min())
# 
# And to transform the scaled feature back to its initial format:
# 
# X_scaled = X_std * (max - min) + min

# In[9]:


# scaling the features between 0 and 1. 

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[10]:


#let's have a look at the scaled training dataset
print('Mean: ', X_train_scaled.mean(axis=0))
print('Standard Deviation: ', X_train_scaled.std(axis=0))
print('Minimum value: ', X_train_scaled.min(axis=0))
print('Maximum value: ', X_train_scaled.max(axis=0))


# # Logistic Regression 

# In[11]:


# model build on unscaled variables

logit = LogisticRegression(random_state=44, C=1000) # c big to avoid regularization
logit.fit(X_train, y_train)
print('Train set')
pred = logit.predict_proba(X_train)
print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = logit.predict_proba(X_test)
print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# In[12]:


logit.coef_


# In[13]:


# model built on scaled variables
logit = LogisticRegression(random_state=44, C=1000) # c big to avoid regularization
logit.fit(X_train_scaled, y_train)
print('Train set')
pred = logit.predict_proba(X_train_scaled)
print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = logit.predict_proba(X_test_scaled)
print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# In[14]:


logit.coef_


# We observe that the performance of logistic regression did not change when using the datasets with the features scaled (compare roc-auc values for train and test set for models with and without feature scaling).
# 
# However, when looking at the coefficients we do see a big difference in the values. This is because the magnitude of the variable was affecting the coefficients. After scaling, all 3 variables have the relative same effect (coefficient) for survival, whereas before scaling, we would be inclined to think that PClass was driving the Survival outcome.

# # Support Vector Machines

# In[15]:


# model build on data with plenty of categories in Cabin variable

SVM_model = SVC(random_state=44, probability=True)
SVM_model.fit(X_train, y_train)
print('Train set')
pred = SVM_model.predict_proba(X_train)
print('SVM roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = SVM_model.predict_proba(X_test)
print('SVM roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# In[16]:


SVM_model = SVC(random_state=44, probability=True)
SVM_model.fit(X_train_scaled, y_train)
print('Train set')
pred = SVM_model.predict_proba(X_train_scaled)
print('SVM roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = SVM_model.predict_proba(X_test_scaled)
print('SVM roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# # Neural Networks

# In[17]:


# model built on unscaled features

NN_model = MLPClassifier(random_state=44, solver='sgd')
NN_model.fit(X_train, y_train)
print('Train set')
pred = NN_model.predict_proba(X_train)
print('Neural Network roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = NN_model.predict_proba(X_test)
print('Neural Network roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# In[18]:


# model built on scaled features

NN_model = MLPClassifier(random_state=44, solver='sgd')
NN_model.fit(X_train_scaled, y_train)
print('Train set')
pred = NN_model.predict_proba(X_train_scaled)
print('Neural Network roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = NN_model.predict_proba(X_test_scaled)
print('Neural Network roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# # K-Nearest Neighbours

# In[19]:


#model built on unscaled features

KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train, y_train)
print('Train set')
pred = KNN.predict_proba(X_train)
print('KNN roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = KNN.predict_proba(X_test)
print('KNN roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# In[20]:


# model built on scaled

KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train_scaled, y_train)
print('Train set')
pred = KNN.predict_proba(X_train_scaled)
print('KNN roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = KNN.predict_proba(X_test_scaled)
print('KNN roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# Both KNN methods are over-fitting to the train set. Thus, we would need to change the parameters of the model or use less features to try and decrease over-fitting, which exceeds the purpose of this demonstration.

# # Random Forests

# In[21]:


# model built on unscaled features

rf = RandomForestClassifier(n_estimators=700, random_state=39)
rf.fit(X_train, y_train)
print('Train set')
pred = rf.predict_proba(X_train)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = rf.predict_proba(X_test)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# In[22]:


# model built in scaled features
rf = RandomForestClassifier(n_estimators=700, random_state=39)
rf.fit(X_train_scaled, y_train)
print('Train set')
pred = rf.predict_proba(X_train_scaled)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = rf.predict_proba(X_test_scaled)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# As expected, Random Forests shows no change in performance regardless of whether it is trained on a dataset with scaled or unscaled features

# In[23]:


ada = AdaBoostClassifier(n_estimators=200, random_state=44)
ada.fit(X_train, y_train)
print('Train set')
pred = ada.predict_proba(X_train)
print('AdaBoost roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = ada.predict_proba(X_test)
print('AdaBoost roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# In[24]:


ada = AdaBoostClassifier(n_estimators=200, random_state=44)
ada.fit(X_train_scaled, y_train)
print('Train set')
pred = ada.predict_proba(X_train_scaled)
print('AdaBoost roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = ada.predict_proba(X_test_scaled)
print('AdaBoost roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# As expected, AdaBoost shows no change in performance regardless of whether it is trained on a dataset with scaled or unscaled features
