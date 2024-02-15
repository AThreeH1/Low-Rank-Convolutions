#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Creating a low rank function f as described in the paper


# In[12]:


import pandas as pd

path = "C:\Study\DS\HyenaVsMHApy\EURUSD"
ColName = ',0,1,2,3,4,5,6'
df = pd.read_csv(path, delimiter = '\t')


# In[13]:


df[['Index','Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']] = df[ColName].str.split(',', expand=True)
df.drop(columns=[ColName, 'Index'], inplace=True)


# In[16]:


df.head()


# In[21]:


columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume']
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce').astype(float)


# In[22]:


nan_count = df.isna().sum()
print(nan_count)


# In[23]:


df.dropna(inplace = True)


# In[32]:


def f1(x):
    return (1/df['Close'][x])

def f2(y):
    return 100*(df['Close'][y])

def f3(x):
    return -1

def f4(y):
    return 100

def function(x, y):
    if y < x:
        return('Error: please select y >= x')
    
    PerChange = f1(x)*f2(y) + f3(x)*f4(y)
    return PerChange


# In[33]:


function(1, 4)


# In[ ]:




