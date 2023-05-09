#!/usr/bin/env python
# coding: utf-8

# In[15]:


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
from ecotools.pi_client import pi_client
import requests
import seaborn as sns 
from sklearn.cluster import KMeans
import os
os.chdir(r"C:\Automation\kBtu Baselines")

import warnings
warnings.filterwarnings('ignore')
#print(mnv.version)  # Last updated for mnv v2.0.1
from dateutil.relativedelta import relativedelta
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.metrics import r2_score
import math
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
import auto_run_ele_tag
import logging
import ecotools.auto


# In[ ]:


pc = pi_client(username= r'dbsorian')
def main():
    from datetime import datetime, timedelta
    #pred_start = pd.to_datetime(datetime.now().strftime('%Y-%m-%d %H:00:00')) - timedelta(days=1)
    #pred_end = datetime.now().strftime('%Y-%m-%d %H:00:00')
    pc = pi_client(username= r'dbsorian')


# In[19]:


data_name = pc.search_by_point("*Electricity*demand_kBtu")
data_name = [ele for ele in data_name] 
pred_start = pd.to_datetime(datetime.now().strftime('%Y-%m-%d %H:00:00')) - timedelta(days=365)
pred_end = datetime.now().strftime('%Y-%m-%d %H:00:00')
# auto_run_ele_tag
appended_data = []
for i in range(len(data_name)):
    appended_data.append(auto_run_ele_tag.auto_run(data_name = data_name[i]))   
    
appended_data = pd.concat(appended_data, axis=1)

send_df = appended_data.copy()
send_df.columns = send_df.columns.str.replace('Electricity_Demand_kBtu','Baseline_Modeled_Electricity')
send_df.dropna(axis=0, how='all', inplace=True)

chunk_split = 24
chunk = int(len(send_df)/chunk_split)

i = 0
while i < chunk_split:
    pc.write_data_to_pi(send_df.iloc[chunk*i:chunk*(i+1), :], update_option='Replace', override='pinkdinosaur')
    i += 1

    
if __name__ == "__main__":
    ecotools.auto.run_task(main, log_name='kbtu_model_ele', recipients=['dsimperiale@ucdavis.edu'])



# In[ ]:





# In[ ]:




