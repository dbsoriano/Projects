
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
# import import_ipynb
import auto_run_steam_tag-Copy1
import logging
import ecotools.auto


def main():
    from datetime import datetime, timedelta
    pc = pi_client()
    
    data_name = pc.search_by_point("*steam*demand_kBtu")

    appended_data = [auto_run(data_name,pred_start=None, pred_end=None)]

    send_df = appended_data.copy()
    send_df.columns = send_df.columns.str.replace('Steam_Demand_kBtu','Baseline_Modeled_Steam')
    
    

if __name__ == "__main__":

    
    ecotools.auto.run_task(main, log_name='kbtu_model_steam', recipients=['dsimperiale@ucdavis.edu'])

# %%
