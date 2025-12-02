#!/usr/bin/env python
# coding: utf-8

# Mixed variables
# Mixed variables are those which values contain both numbers and labels.
# 
# Variables can be mixed for a variety of reasons. For example, when credit agencies gather and store financial information of users, usually, the values of the variables they store are numbers. However, in some cases the credit agencies cannot retrieve information for a certain user for different reasons. What Credit Agencies do in these situations is to code each different reason due to which they failed to retrieve information with a different code or 'label'. Like this, they generate mixed type variables. These variables contain numbers when the value could be retrieved, or labels otherwise.
# 
# As an example, think of the variable 'number_of_open_accounts'. It can take any number, representing the number of different financial accounts of the borrower. Sometimes, information may not be available for a certain borrower, for a variety of reasons. Each reason will be coded by a different letter, for example: 'A': couldn't identify the person, 'B': no relevant data, 'C': person seems not to have any open account.
# 
# Another example of mixed type variables, is for example the variable missed_payment_status. This variable indicates, whether a borrower has missed a (any) payment in their financial item. For example, if the borrower has a credit card, this variable indicates whether they missed a monthly payment on it. Therefore, this variable can take values of 0, 1, 2, 3 meaning that the customer has missed 0-3 payments in their account. And it can also take the value D, if the customer defaulted on that account.
# 
# Typically, once the customer has missed 3 payments, the lender declares the item defaulted (D), that is why this variable takes numerical values 0-3 and then D.
# 
# For this lecture, you will need to download a toy csv file that I created and uploaded at the end of the lecture in Udemy. It is called toy.csv.

# In[1]:


import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# open_il_24m indicates:
# "Number of installment accounts opened in past 24 months".
# Installment accounts are those that, at the moment of acquiring them,
# there is a set period and amount of repayments agreed between the
# lender and borrower. An example of this is a car loan, or a student loan.
# the borrowers know that they are going to pay a certain,
# fixed amount over, for example 36 months.

data = pd.read_csv('C:/Users/sidda/Downloads/toy.csv')
data.head()


# In[3]:


data.shape


# In[4]:


# 'A': couldn't identify the person
# 'B': no relevant data
# 'C': person seems not to have any account open

data.Age.unique()


# In[5]:


# Now, let's make a bar plot showing the different number of 
# borrowers for each of the values of the mixed variable

fig = data.Age.value_counts().plot.bar()
fig.set_title('Number of installment accounts open')
fig.set_ylabel('Number of borrowers')


# In[ ]:




