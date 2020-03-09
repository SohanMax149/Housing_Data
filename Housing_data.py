#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# In[2]:


x=pd.read_csv(r'C:\ML\Housing\Housingdata.csv')


# In[3]:


x.head()


# In[4]:


x.describe()


# In[5]:


x.isnull().sum()


# In[6]:


x.dtypes


# In[7]:


y=x.MEDV 
x=x.drop(columns=['MEDV'],axis=1)


# In[8]:


x.head()


# In[9]:


y.head()


# In[10]:


CRIM_median_train=x.CRIM.median()
x.CRIM=x.CRIM.fillna(CRIM_median_train)
print(CRIM_median_train)


# In[11]:


ZN_median_train=x.ZN.median()
x.ZN=x.ZN.fillna(ZN_median_train)
print(ZN_median_train)


# In[12]:


INDUS_median_train=x.INDUS.median()
x.INDUS=x.INDUS.fillna(INDUS_median_train)
print(INDUS_median_train)


# In[13]:


x.isnull().sum()


# In[14]:


x['CHAS']=x['CHAS'].fillna(0)


# In[15]:


x.isnull().sum()


# In[16]:


AGE_median_train=x.AGE.median()
x.AGE=x.AGE.fillna(AGE_median_train)
print(AGE_median_train)


# In[17]:


x.isnull().sum()


# In[18]:


LSTAT_median_train=x.LSTAT.median()
x.LSTAT=x.LSTAT.fillna(LSTAT_median_train)
print(LSTAT_median_train)


# In[19]:


x.isnull().sum()


# In[20]:


corr_matrix=x.corr().round(1)
plt.figure(figsize=(15,10))
sns.heatmap(data=corr_matrix, annot=True, linewidths=0.1, square=True)


# In[21]:


x['TAXRAD']= x['TAX']+x['RAD']
x=x.drop(['TAX','RAD','NOX'], axis=1)


# In[22]:


x.hist(bins=10,figsize=(12,9),grid=False)


# In[23]:


x['AGE']=np.log(x['AGE'])


# In[24]:


x['B']=np.log(x['B'])


# In[25]:


x['DIS']=np.log(x['DIS'])


# In[26]:


x['LSTAT']=np.log(x['LSTAT'])


# In[27]:


x['TAXRAD']=np.log(x['TAXRAD'])


# In[28]:


x.hist(bins=10,figsize=(12,9),grid=False)


# In[29]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[30]:


x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.3,random_state=100)


# In[31]:


coef=LinearRegression()
coef.fit(x_train, y_train)


# In[32]:


Y_pred = coef.predict(x_test)


# In[33]:


print(Y_pred)


# In[34]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(coef,x,y)
print(scores)


# In[35]:


print('Variance score: %.2f' % r2_score(y_test, Y_pred))


# In[36]:


from math import sqrt


# In[37]:


rmse=sqrt(mean_squared_error(y_test, Y_pred))


# In[38]:


print(rmse)


# In[39]:


sns.distplot(x.CRIM)


# In[40]:


sns.distplot(x.PTRATIO)


# In[41]:


y.describe()


# In[ ]:





# In[ ]:




