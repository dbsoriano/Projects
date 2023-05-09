#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import linear_model 
import matplotlib.pyplot as plt
import seaborn as sns
from ecotools.pi_client import pi_client
pi = pi_client(username= r'dbsorian')
import numpy as np


# In[7]:


import sys
sys.path


# In[8]:


tags = pi.search_by_point(['*Ghausi*chilledWater*kBtu', 'aiTIT4045'])

print(tags)


# In[9]:


data


# In[10]:


start = '2019-09-28 10:00:00'
end = '2021-09-28 10:00:00'    


calc = 'interpolated' 
interval = '1h'   

# Extras
chunk_size = 40
weight = 'TimeWeighted'
summary_calc = 'average'
max_count = round(1500000/len(tags))

data = pi.get_stream_by_point(tags, start=start, end=end, _convert_cols='numeric', calculation=calc, interval=interval, _weight=weight, _summary_type=summary_calc, _max_count=max_count, _chunk_size=chunk_size)


# In[11]:


df = data.copy()


# In[12]:


df.head(10)
df.dtypes


# In[13]:


print(df.shape)


# In[14]:


df_ow = df.loc[:,"Ghausi_ChilledWater_Demand_kBtu"]


# In[15]:


df_ow.plot()


# In[16]:


df['Ghausi_ChilledWater_Demand_kBtu'] = df['Ghausi_ChilledWater_Demand_kBtu'].fillna(df['Ghausi_ChilledWater_Demand_kBtu'].mean())
print(df.isnull().sum())


# In[32]:


df_x = df.drop(['aiTIT4045'], axis = 1)
df_y = df['Ghausi_ChilledWater_Demand_kBtu']


# In[36]:


from sklearn.model_selection import train_test_split as tts
train_df_x, test_df_x, train_df_y, test_df_y = tts(df_x, df_y, random_state = 42, test_size = 0.3)


# In[37]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_df_x, train_df_y)

score = model.score(test_df_x, test_df_y)
print(f'R^2 score: {score:.3f}')


# In[38]:


from sklearn.metrics import mean_squared_error, r2_score

y_prediction = model.predict(test_df_x)
                             
print('MSE :', mean_squared_error(test_df_y, y_prediction))
print('Coefficient of determination:', r2_score(test_df_y, y_prediction))


# In[42]:


residuals = test_df_y - y_prediction
sns.residplot(x=y_prediction, y=residuals, lowess=True, color='g')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()


# In[ ]:




