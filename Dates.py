#!/usr/bin/env python
# coding: utf-8

# Dates and Times A special type of categorical variable are those that instead of taking traditional labels, like color (blue, red), or city (London, Manchester), take dates as values. For example, date of birth ('29-08-1987', '12-01-2012'), or time of application ('2016-Dec', '2013-March').
# 
# Datetime variables can contain dates only, or time only, or date and time.
# 
# Typically, we would never work with a date variable as a categorical variable, for a variety of reasons:
# 
# Date variables usually contain a huge number of individual categories, which will expand the feature space dramatically Date variables allow us to capture much more information from the dataset if preprocessed in the right way In addition, often, date variables will contain dates that were not present in the dataset that we used to train the machine learning algorithm. In fact, will contain dates placed in the future respect to the dates present in the dataset we used to train. Therefore, the machine learning model will not know what to do with them, because it never saw them while being trained.

# Dates and Times A special type of categorical variable are those that instead of taking traditional labels, like color (blue, red), or city (London, Manchester), take dates as values. For example, date of birth ('29-08-1987', '12-01-2012'), or time of application ('2016-Dec', '2013-March').
# 
# Datetime variables can contain dates only, or time only, or date and time.
# 
# Typically, we would never work with a date variable as a categorical variable, for a variety of reasons:
# 
# Date variables usually contain a huge number of individual categories, which will expand the feature space dramatically Date variables allow us to capture much more information from the dataset if preprocessed in the right way In addition, often, date variables will contain dates that were not present in the dataset that we used to train the machine learning algorithm. In fact, will contain dates placed in the future respect to the dates present in the dataset we used to train. Therefore, the machine learning model will not know what to do with them, because it never saw them while being trained.
# 
# I will cover different was of pre-processing/engineering date variables in the section "Engineering date variables".
# 
# =============================================================================
# 
# Real Life example: Peer to peer lending (Finance) Lending Club Lending Club is a peer-to-peer Lending company based in the US. They match people looking to invest money with people looking to borrow money. When investors invest their money through Lending Club, this money is passed onto borrowers, and when borrowers pay their loans back, the capital plus the interest passes on back to the investors. It is a win for everybody as they can get typically lower loan rates and higher investor returns.
# 
# If you want to learn more about Lending Club follow this link.
# 
# The Lending Club dataset contains complete loan data for all loans issued through 2007-2015, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. Features include credit scores, number of finance inquiries, address including zip codes and state, and collections among others. Collections indicates whether the customer has missed one or more payments and the team is trying to recover their money.
# 
# The file is a matrix of about 890 thousand observations and 75 variables. More detail on this dataset can be found in Kaggle's website
# 
# Let's go ahead and have a look at the variables!

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# let's load the Lending Club dataset with a few selected columns

use_cols = ['loan_amnt', 'grade', 'purpose', 'issue_d', 'last_pymnt_d']

data = pd.read_csv('C:/Users/sidda/Downloads/loan.csv', usecols=use_cols)

data.head()


# In[3]:


data.dtypes


# In[4]:


# now let's parse the dates, currently coded as strings, into datetime format
# this will allow us to make some analysis afterwards

data['issue_dt'] = pd.to_datetime(data.issue_d)
data['last_pymnt_dt'] = pd.to_datetime(data.last_pymnt_d)

data[['issue_d', 'issue_dt', 'last_pymnt_d', 'last_pymnt_dt']].head()


# In[5]:


# let's see how much money Lending Club has disbursed
# (i.e., lent) over the years to the different risk
# markets (grade variable)

fig = data.groupby(['issue_dt', 'grade'])['loan_amnt'].sum().unstack().plot(
    figsize=(14, 8), linewidth=2)

fig.set_title('Disbursed amount in time')
fig.set_ylabel('Disbursed Amount (US Dollars)')


# Lending Club seems to have increased the amount of money lent from 2013 onwards. The tendency indicates that they continue to grow. In addition, we can see that their major business comes from lending money to C and B grades.
# 
# 'A' grades are the lower risk borrowers, this is borrowers that most likely will be able to repay their loans, as they are typically in a better financial situation. Borrowers within this grade are typically charged lower interest rates.
# 
# E, F and G grades represent the riskier borrowers. Usually borrowers in somewhat tighter financial situations, or for whom there is not sufficient financial history to make a reliable credit assessment. They are typically charged higher rates, as the business, and therefore the investors, take a higher risk when lending them money.
# 
# Lending Club lends the biggest fraction to borrowers that intend to use that money to repay other debt or credit cards.
