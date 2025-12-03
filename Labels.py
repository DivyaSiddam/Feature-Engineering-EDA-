#!/usr/bin/env python
# coding: utf-8

# Labels
# The values of a categorical variable are selected from a group of categories, also called labels. Thus, in the variable gender the categories or labels are male and female, whereas in the variable city the labels can be London, Manchester, Brighton and so on.
# 
# We can see from the above examples, that different categorical variables can contain different amount of labels (or categories). The variable gender contains only 2 labels, but a variable like city or postcode, can contain a huge number of different labels.
# 
# The number of labels within a categorical variable is known as cardinality. A high number of labels within a variable is known as high cardinality.
# 
# Are multiple labels in a categorical variable a problem?
# If highly cardinal categorical variables are used to train machine learning models, the following problems may appear:
# 
# Variables with too many labels tend to dominate over those with only a few labels, particularly in Tree based algorithms.
# 
# A big number of labels within a variable may introduce noise with little if any information, therefore making the machine learning models prone to over-fit.
# 
# Some of the labels may only be present in the training data set, but not in the test set, therefore causing the machine learning algorithms to over-fit the training set.
# 
# Contrarily, new labels may appear in the test set that were not present in the training set, therefore leaving the machine learning algorithm unable to perform a calculation over the new observation.
# 
# In particular, tree methods are biased towards variables with lots of labels, so their performance may be affected by these type of variables.
# 
# Below, I will show the effect of high cardinality of variables on the performance of different machine learning algorithms, and how a quick fix to reduce the number of labels, without any sort of data insight, already helps to boost performance.
# 
# Real Life example:
# Predicting Survival on the Titanic: understanding society behaviour and beliefs
# Perhaps one of the most infamous shipwrecks in history, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 people on board. Interestingly, by analysing the probability of survival based on few attributes like gender, age, and social status, we can make very accurate predictions on which passengers would survive. Some groups of people were more likely to survive than others, such as women, children, and the upper-class. Therefore, we can learn about the society priorities and privileges at the time.
# 
# ====================================================================================================
# 
# To download the Titanic data, go ahead to this website
# 
# Click on the link 'train.csv', and then click the 'download' blue button towards the right of the screen, to download the dataset. Save it in a folder of your choice.
# 
# Note that you need to be logged in to Kaggle in order to download the datasets.
# 
# If you save it in the same directory from which you are running this notebook, and you rename the file to 'titanic.csv' then you can load it the same way I will load it below.

# In[6]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# In[7]:


# let's load the titanic dataset

data = pd.read_csv('C:/Users/sidda/Downloads/titanic.csv')
data.head()


# The categorical variables in this dataset are Name, Sex, Ticket, Cabin and Embarked.

# In[8]:


# let's inspect at the number of labels for the different categorical variables

print('Number of categories in the variable Name: {}'.format(
    len(data.Name.unique())))

print('Number of categories in the variable Gender: {}'.format(
    len(data.Sex.unique())))

print('Number of categories in the variable Ticket: {}'.format(
    len(data.Ticket.unique())))

print('Number of categories in the variable Cabin: {}'.format(
    len(data.Cabin.unique())))

print('Number of categories in the variable Embarked: {}'.format(
    len(data.Embarked.unique())))

print('Total number of passengers in the Titanic: {}'.format(len(data)))


# While the variable Sex contains only 2 categories and Embarked 3, the variables names, Ticket and Cabin, as expected, contain a huge number of different labels. They show high cardinality. And so does the variable Name (different for each passenger).
# 
# To demonstrate the effect of high cardinality in machine learning performance, I will work with the variable Cabin.

# In[9]:


# let's explore the values / categories of Cabin

# we know from the previous cell that there are 148 different cabins
# therefore the variable is highly cardinal

data.Cabin.unique()


# Let's now reduce the cardinality of the variable. How? instead of using the entire Cabin value, I will capture only the first letter.
# 
# Rationale: the first letter indicates the deck on which the cabin was located, and is therefore an indication of both social class status and proximity to the surface of the Titanic. Both are known to help or improve the probability of survival.

# In[10]:


# let's capture the first letter
data['Cabin_reduced'] = data['Cabin'].astype(str).str[0]

data[['Cabin', 'Cabin_reduced']].head()


# In[11]:


print('Number of categories in the variable Cabin: {}'.format(
    len(data.Cabin.unique())))

print('Number of categories in the variable Cabin reduced: {}'.format(
    len(data.Cabin_reduced.unique())))


# We reduced the number of different labels from 148 to 8.

# In[12]:


# let's separate into training and testing set
# in order to build machine learning models

use_cols = ['Cabin', 'Cabin_reduced', 'Sex']

X_train, X_test, y_train, y_test = train_test_split(
    data[use_cols], 
    data.Survived,  
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape


# # High cardinality leads to uneven distribution of categories in train and test sets

# When a variable is highly cardinal, often some categories land only on the training set, or only on the testing set. If present only in the training set, they may lead to overfitting. If present only on the testing set, the machine learning algorithm will not know how to handle them, as it has not seen it during training.

# In[13]:


# Let's find out labels present only in the training set

unique_to_train_set = [
    x for x in X_train.Cabin.unique() if x not in X_test.Cabin.unique()
]

len(unique_to_train_set)


# There are 49 Cabins only present in the training set, and not in the testing set.

# In[14]:


# Let's find out labels present only in the test set

unique_to_test_set = [
    x for x in X_test.Cabin.unique() if x not in X_train.Cabin.unique()
]

len(unique_to_test_set)


# Variables with high cardinality tend to have values (i.e., categories) present in the training set, that are not present in the test set, and vice versa. This will bring problems at the time of training (due to over-fitting) and scoring of new data (how should the model deal with unseen categories?).
# 
# In order to evaluate the effect of categorical variables in machine learning models, I will quickly replace the categories by numbers. See below.

# In[15]:


# Let's re-map Cabin into numbers so we can use it to train ML models

# I will replace each cabin by a number
# This will allow me to quickly demonstrate
# the effect of labels on machine learning algorithms

cabin_dict = {k:i for i, k in enumerate(X_train.Cabin.unique(), 0)} 
cabin_dict


# In[16]:


# replace the labels in Cabin, using the dic created above
X_train.loc[:, 'Cabin_mapped'] = X_train.loc[:, 'Cabin'].map(cabin_dict)
X_test.loc[:, 'Cabin_mapped'] = X_test.loc[:, 'Cabin'].map(cabin_dict)

X_train[['Cabin_mapped', 'Cabin']].head(10)


# We see how NaN takes the value 1 in the new variable, C46 takes the value 0, B71 takes the value 2, and so on.

# In[17]:


# Now I will replace the letters in the reduced cabin variable

# create replace dictionary
cabin_dict = {k: i for i, k in enumerate(X_train['Cabin_reduced'].unique(), 0)}

# replace labels by numbers with dictionary
X_train.loc[:, 'Cabin_reduced'] = X_train.loc[:, 'Cabin_reduced'].map(cabin_dict)
X_test.loc[:, 'Cabin_reduced'] = X_test.loc[:, 'Cabin_reduced'].map(cabin_dict)

X_train[['Cabin_reduced', 'Cabin']].head(10)


# We see now that NAN and  take the same number, 1, because we are capturing only the letter. They both start with D.

# In[18]:


# re-map the categorical variable Sex into numbers

X_train.loc[:, 'Sex'] = X_train.loc[:, 'Sex'].map({'male': 0, 'female': 1})
X_test.loc[:, 'Sex'] = X_test.loc[:, 'Sex'].map({'male': 0, 'female': 1})

X_train.Sex.head()


# In[19]:


# check if there are missing values in these variables

X_train[['Cabin_mapped','Cabin_reduced', 'Sex']].isnull().sum()


# In[20]:


X_test[['Cabin_mapped','Cabin_reduced', 'Sex']].isnull().sum()


# In the test set, there are now 30 missing values for the highly cardinal variable. These were introduced when encoding the categories into numbers. How? Many categories exist only in the test set. Thus, when we creating our encoding dictionary using only the train set, we did not generate a number to replace those labels present only in the test set. As a consequence, they were encoded as NaN. We will see in future lectures how to tackle this problem. For now, I will fill those missing values with 0.

# In[21]:


# let's check the number of categories in the encoded variables
len(X_train.Cabin_mapped.unique()), len(X_train.Cabin_reduced.unique())


# We can see how we reduced the number of different categories from 148 to just 9 in our previous step. Let's go ahead and evaluate the effect of labels in machine learning algorithms.

# # Random Forests

# In[23]:


# model built on data with high cardinality for cabin

# call the model
rf = RandomForestClassifier(n_estimators=200, random_state=39)

# train the model
rf.fit(X_train[['Cabin_mapped', 'Sex']], y_train)

# make predictions on train and test set
pred_train = rf.predict_proba(X_train[['Cabin_mapped', 'Sex']])
pred_test = rf.predict_proba(X_test[['Cabin_mapped', 'Sex']].fillna(0))

print('Train set')
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))


# We observe that the performance of the Random Forests on the training set is quite superior to its performance in the test set. This indicates that the model is over-fitting, which means that it does a great job at predicting the outcome on the dataset it was trained on, but it lacks the power to generalise the prediction to unseen data.

# In[24]:


# model built on data with low cardinality for cabin

# call the model
rf = RandomForestClassifier(n_estimators=200, random_state=39)

# train the model
rf.fit(X_train[['Cabin_reduced', 'Sex']], y_train)

# make predictions on train and test set
pred_train = rf.predict_proba(X_train[['Cabin_reduced', 'Sex']])
pred_test = rf.predict_proba(X_test[['Cabin_reduced', 'Sex']])

print('Train set')
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))


# We can see now that the Random Forests no longer over-fit to the training set. In addition, the model is much better at generalising the predictions (compare the roc-auc of this model on the test set vs the roc-auc of the model above also in the test set: 0.83 vs 0.80).

# # AdaBoost
# 

# In[8]:


# ---------------------------------------------
# 1. IMPORTS
# ---------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score

# ---------------------------------------------
# 2. LOAD YOUR DATA
# ---------------------------------------------
# IMPORTANT: Replace the file name below with YOUR actual CSV file
df = pd.read_csv("C:/Users/sidda/Downloads/titanic.csv")   

# ---------------------------------------------
# 3. PREPARE FEATURES
# ---------------------------------------------
# Convert Sex to numeric
if df['Sex'].dtype == 'object':
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Create Cabin_mapped from Cabin
if 'Cabin_mapped' not in df.columns:
    df['Cabin_mapped'] = df['Cabin'].astype(str).str[0].map({
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8
    }).fillna(0).astype(int)

# ---------------------------------------------
# 4. SELECT X AND Y
# ---------------------------------------------
X = df[['Cabin_mapped', 'Sex']]
y = df['Survived']

# ---------------------------------------------
# 5. TRAIN/TEST SPLIT
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=44
)

# ---------------------------------------------
# 6. ADABOOST MODEL
# ---------------------------------------------
ada = AdaBoostClassifier(n_estimators=200, random_state=44)

# Fit the model
ada.fit(X_train, y_train)

# Predictions
pred_train = ada.predict_proba(X_train)[:,1]
pred_test = ada.predict_proba(X_test.fillna(0))[:,1]

# ---------------------------------------------
# 7. RESULTS
# ---------------------------------------------
print("Train AUC:", roc_auc_score(y_train, pred_train))
print("Test AUC:", roc_auc_score(y_test, pred_test))


# In[9]:


# model build on data with plenty of categories in Cabin variable

# call the model
ada = AdaBoostClassifier(n_estimators=200, random_state=44)

# train the model
ada.fit(X_train[['Cabin_mapped', 'Sex']], y_train)

# make predictions on train and test set
pred_train = ada.predict_proba(X_train[['Cabin_mapped', 'Sex']])
pred_test = ada.predict_proba(X_test[['Cabin_mapped', 'Sex']].fillna(0))

print('Train set')
print('Adaboost roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Adaboost roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))


# In[12]:


# ---------------------------------------------
# 1. IMPORTS
# ---------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score

# ---------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------
# CHANGE THIS to your actual dataset filename
df = pd.read_csv("C:/Users/sidda/Downloads/titanic.csv")   # <<< Replace with your file

# ---------------------------------------------
# 3. PREPARE FEATURES
# ---------------------------------------------
# Convert Sex to numeric (male=0, female=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Create Cabin_mapped column if it doesn't exist
df['Cabin_mapped'] = df['Cabin'].astype(str).str[0].map({
    'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8
}).fillna(0).astype(int)

# ---------------------------------------------
# 4. SELECT FEATURES AND TARGET
# ---------------------------------------------
X = df[['Cabin_mapped', 'Sex']]
y = df['Survived']

# ---------------------------------------------
# 5. TRAIN-TEST SPLIT
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=44
)

# ---------------------------------------------
# 6. BUILD AND TRAIN ADABOOST MODEL
# ---------------------------------------------
ada = AdaBoostClassifier(n_estimators=200, random_state=44)

# Train
ada.fit(X_train[['Cabin_mapped', 'Sex']], y_train)

# Predictions
pred_train = ada.predict_proba(X_train[['Cabin_mapped', 'Sex']])[:,1]
pred_test = ada.predict_proba(X_test[['Cabin_mapped', 'Sex']].fillna(0))[:,1]

# ---------------------------------------------
# 7. EVALUATE
# ---------------------------------------------
print('Train set')
print('Adaboost roc-auc: {}'.format(roc_auc_score(y_train, pred_train)))
print('Test set')
print('Adaboost roc-auc: {}'.format(roc_auc_score(y_test, pred_test)))


# Similarly, the Adaboost model trained on the variable with high cardinality is slightly overfit to the train set. Whereas the Adaboost trained on the low cardinal variable is not overfitting and it does a better job in generalising the predictions.
# 
# In addition, building an AdaBoost on a model with less categories in Cabin, is a) simpler and b) should a different category in the test set appear, by taking just the front letter of cabin, the ML model will know how to handle it because it was seen during training.

# # Logistic Regression

# In[15]:


from sklearn.linear_model import LogisticRegression

# model build on data with plenty of categories in Cabin variable

# call the model
logit = LogisticRegression(random_state=44)

# train the model
logit.fit(X_train[['Cabin_mapped', 'Sex']], y_train)

# make predictions on train and test set
pred_train = logit.predict_proba(X_train[['Cabin_mapped', 'Sex']])
pred_test = logit.predict_proba(X_test[['Cabin_mapped', 'Sex']].fillna(0))

print('Train set')
print('Logistic regression roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Logistic regression roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))


# In[ ]:


# ---------------------------------------------
# 1. IMPORTS
# ---------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score

# ---------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------
# Replace with your actual CSV filename
df = pd.read_csv("C:/Users/sidda/Downloads/titanic.csv")  # <<< CHANGE THIS

# ---------------------------------------------
# 3. PREPARE FEATURES
# ---------------------------------------------
# Convert Sex to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Create Cabin_reduced column
df['Cabin_reduced'] = df['Cabin'].astype(str).str[0].map({
    'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8
}).fillna(0).astype(int)

# ---------------------------------------------
# 4. SELECT FEATURES AND TARGET
# ---------------------------------------------
X = df[['Cabin_reduced', 'Sex']]
y = df['Survived']

# ---------------------------------------------
# 5. TRAIN-TEST SPLIT
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=44
)

# ---------------------------------------------
# 6. LOGISTIC REGRESSION
# ---------------------------------------------
logit = LogisticRegression(random_state=44)
logit.fit(X_train[['Cabin_reduced', 'Sex']], y_train)

pred_train_logit = logit.predict_proba(X_train[['Cabin_reduced', 'Sex']])[:,1]
pred_test_logit = logit.predict_proba(X_test[['Cabin_reduced', 'Sex']].fillna(0))[:,1]

print('--- Logistic Regression ---')
print('Train AUC:', roc_auc_score(y_train, pred_train_logit))
print('Test AUC:', roc_auc_score(y_test, pred_test_logit))

# ---------------------------------------------
# 7. ADABOOST
# ---------------------------------------------
ada = AdaBoostClassifier(n_estimators=200, random_state=44)
ada.fit(X_train[['Cabin_reduced', 'Sex']], y_train)

pred_train_ada = ada.predict_proba(X_train[['Cabin_reduced', 'Sex']])[:,1]
pred_test_ada = ada.predict_proba(X_test[['Cabin_reduced', 'Sex']].fillna(0))[:,1]

print('--- AdaBoost ---')
print('Train AUC:', roc_auc_score(y_train, pred_train_ada))
print('Test AUC:', roc_auc_score(y_test, pred_test_ada))


# In[19]:


# model build on data with fewer categories in Cabin Variable

# call the model
logit = LogisticRegression(random_state=44)

# train the model
logit.fit(X_train[['Cabin_reduced', 'Sex']], y_train)

# make predictions on train and test set
pred_train = logit.predict_proba(X_train[['Cabin_reduced', 'Sex']])
pred_test = logit.predict_proba(X_test[['Cabin_reduced', 'Sex']].fillna(0))

print('Train set')
print('Logistic regression roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Logistic regression roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))


# We can draw the same conclusion for Logistic Regression. Reducing the cardinality improves the performance and generalisation of the algorithm.

# # Gradient Boosted Classifier

# In[22]:


# -------------------------------
# 1. IMPORTS
# -------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# -------------------------------
# 2. LOAD DATA
# -------------------------------
# Replace with your CSV file
df = pd.read_csv("C:/Users/sidda/Downloads/titanic.csv")  # <<< CHANGE THIS

# -------------------------------
# 3. PREPARE FEATURES
# -------------------------------
# Convert Sex to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Create Cabin_mapped column for plenty of categories
df['Cabin_mapped'] = df['Cabin'].astype(str).str[0].map({
    'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8
}).fillna(0).astype(int)

# -------------------------------
# 4. SELECT FEATURES AND TARGET
# -------------------------------
X = df[['Cabin_mapped', 'Sex']]
y = df['Survived']

# -------------------------------
# 5. TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=44
)

# -------------------------------
# 6. GRADIENT BOOSTED TREES
# -------------------------------
gbc = GradientBoostingClassifier(n_estimators=300, random_state=44)

# Train the model
gbc.fit(X_train[['Cabin_mapped', 'Sex']], y_train)

# Predictions
pred_train = gbc.predict_proba(X_train[['Cabin_mapped', 'Sex']])[:,1]
pred_test = gbc.predict_proba(X_test[['Cabin_mapped', 'Sex']].fillna(0))[:,1]

# -------------------------------
# 7. EVALUATE
# -------------------------------
print('Train set')
print('Gradient Boosted Trees roc-auc: {}'.format(roc_auc_score(y_train, pred_train)))
print('Test set')
print('Gradient Boosted Trees roc-auc: {}'.format(roc_auc_score(y_test, pred_test)))


# In[23]:


# model build on data with plenty of categories in Cabin variable

# call the model
gbc = GradientBoostingClassifier(n_estimators=300, random_state=44)

# train the model
gbc.fit(X_train[['Cabin_mapped', 'Sex']], y_train)

# make predictions on train and test set
pred_train = gbc.predict_proba(X_train[['Cabin_mapped', 'Sex']])
pred_test = gbc.predict_proba(X_test[['Cabin_mapped', 'Sex']].fillna(0))

print('Train set')
print('Gradient Boosted Trees roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Gradient Boosted Trees roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))


# In[25]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# Load your data
df = pd.read_csv("C:/Users/sidda/Downloads/titanic.csv")  # <<< Replace with your file

# Convert Sex to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Create Cabin_reduced column
df['Cabin_reduced'] = df['Cabin'].astype(str).str[0].map({
    'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8
}).fillna(0).astype(int)

# Select features and target
X = df[['Cabin_reduced', 'Sex']]  # MUST include Cabin_reduced
y = df['Survived']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=44
)

# Gradient Boosted Trees
gbc = GradientBoostingClassifier(n_estimators=300, random_state=44)
gbc.fit(X_train[['Cabin_reduced', 'Sex']], y_train)

# Predictions
pred_train = gbc.predict_proba(X_train[['Cabin_reduced', 'Sex']])[:,1]
pred_test = gbc.predict_proba(X_test[['Cabin_reduced', 'Sex']].fillna(0))[:,1]

# Evaluate
print('Train set')
print('Gradient Boosted Trees roc-auc:', roc_auc_score(y_train, pred_train))
print('Test set')
print('Gradient Boosted Trees roc-auc:', roc_auc_score(y_test, pred_test))


# In[26]:


# model build on data with plenty of categories in Cabin variable

# call the model
gbc = GradientBoostingClassifier(n_estimators=300, random_state=44)

# train the model
gbc.fit(X_train[['Cabin_reduced', 'Sex']], y_train)

# make predictions on train and test set
pred_train = gbc.predict_proba(X_train[['Cabin_reduced', 'Sex']])
pred_test = gbc.predict_proba(X_test[['Cabin_reduced', 'Sex']].fillna(0))

print('Train set')
print('Gradient Boosted Trees roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Gradient Boosted Trees roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))


# Gradient Boosted trees are indeed over-fitting to the training set in those cases where the variable Cabin has a lot of labels. This was expected as tree methods tend to be biased to variables with plenty of categories.
