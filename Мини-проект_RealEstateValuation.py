#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_excel('D:\Real_estate_valuation_data_set.xlsx')


# In[4]:


df


# In[27]:


missing_values = df.isnull().sum()


# In[28]:


missing_values


# In[7]:


print(df.dtypes)


# In[8]:


df.rename(columns={'X1 transaction date':'Transaction date', 'X2 house age':'House age',
                   'X3 distance to the nearest MRT station':'Distance to the nearest MRT station',
                   'X4 number of convenience stores':'Number of convenience stores',
                   'X5 latitude':'Latitude', 'X6 longitude':'Longitude', 
                  'Y house price of unit area':'House price of unit area'})


# In[9]:


features = list(df.columns.drop(labels=['X1 transaction date','X5 latitude','X6 longitude','Y house price of unit area']))
features


# In[10]:


X = df[features]
X


# In[11]:


y = df['Y house price of unit area']
y


# In[12]:


LinReg_model = LinearRegression()


# In[13]:


LinReg_model.fit(X.values,y)
LinReg_model.score(X.values,y)


# In[14]:


predicted = LinReg_model.predict([[10,905,2]])
predicted


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Number of convenience stores')
plt.ylabel('House price of unit area')
plt.scatter(df['X4 number of convenience stores'], df['Y house price of unit area'], color='blue', marker='*')


# In[35]:


plt.xlabel('House age')
plt.ylabel('House price of unit area')
plt.scatter(df['X2 house age'], df['Y house price of unit area'], color='green', marker='o')
#plt.plot(df['X2 house age'], df['Y house price of unit area'], color='yellow')


# In[63]:


import seaborn as sns
sns.regplot(x=df['X2 house age'],
            y=df['Y house price of unit area'],
            scatter_kws={'color': 'green'}, 
            line_kws={'color': 'blue', 'linewidth':4.5})

# Настройка графика

# Отображение графика
plt.show()


# In[17]:


df.shape


# In[18]:


coefficients = LinReg_model.coef_
coefficients #  коэффициент(ы) k: y = kx+b; но у нас тут несколько признаков (фичей), поэтому и коэффов несколько
# сколько признаков, столько и коэффициентов:
# например: 2482 признаков - 2482 коэффициентов и т.д.


# In[19]:


LinReg_model.intercept_ #  коэффициент b
# то есть уравнение такое: 
# y = w0 + w1x1 + w2x2 + w3x3, или:
# y = 42.97 - 0.25x1 - 0.005x2 + 1.29x3


# In[60]:


# проходимся по циклу для вывода всех обозначений коэффициентов (весов) и их значений соответственно
for i, coef in enumerate(coefficients, start=1):
    print('Coefficient {} is for w{}'.format(coef, i))


# In[26]:


plt.plot(df['X4 number of convenience stores'], LinReg_model.predict(df[['X4 number of convenience stores','X5 latitude','X6 longitude']].values),
         color='red')


# In[22]:


# как алгоритм (библиотека) считает:


# In[23]:


-0.25285582658775596*10-0.005379129623944099*905+1.297442476101934*2 + 42.9772862060641


# In[24]:


38.175500582721 == predicted
# predicted = LinReg_model.predict([[10,905,2]])


# In[25]:


#LinReg_model.score()

