
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
#os.chdir(r"/Users/danielsoriano/Desktop/ds/kBtu Baselines")

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
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
#import auto_run_steam_tag-Copy1
import logging
import ecotools.auto


import pandas as pd
import numpy as np
from ecotools.pi_client import pi_client
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')



pc = pi_client()

    
# 1. split_data: training and validaton spliting
def split_data(X, y, test_size=0.2, shuffle=False, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size,
                                                      shuffle=shuffle,
                                                      random_state=random_state)
    return X_train, X_val, y_train, y_val

data_name = pc.search_by_point("*Steam*demand_kBtu")


# ## 2. data_manipu: split the cleaned data

# In[5]:


def data_manipu(data_name, df, temp):
    df["Temp"] = temp
    df['cdd'] = 0
    df.loc[df.Temp > 65, 'cdd'] = df.loc[df.Temp > 65, 'Temp'] - 65
    df['hdd'] = 0
    df.loc[df.Temp < 65, 'hdd'] = 65 - df.loc[df.Temp < 65, 'Temp']
    df['cdd2'] = df.cdd**2
    df['hdd2'] = df.hdd**2
    df["Hour"] = df.index.hour
    df["Month"] = df.index.month
    df["DOW"] = df.index.dayofweek
    df['covid'] = 0
    df.loc["2020-03-17":"2021-09-28", 'covid'] = 1
    
    # specifiy the rolling std method
#     df = df.dropna()
#     r = df.rolling(window=20)  # Create a rolling object (no computation yet)
#     mps1 = r.mean() + 3 * r.std()  # Combine a mean and stdev on that object
#     mps2 = r.mean() - 3 * r.std()
#     mps3 = r.mean() + 3 * r.std()
#     mps4 = r.mean() - 3 * r.std()
    
#     # identify outliers
#     cc1=df[df[data_name] > mps1[data_name]]
#     cc2= df[df[data_name] < mps2[data_name]]
#     dd1=df[df.Temp > mps3.Temp]
#     dd2=df[df.Temp < mps4.Temp]

#     # remove outliers
#     index_names_1 = cc1.index_names_1 = cc1[data_name].index
#     index_names_2 = cc2.index_names_2 = cc2[data_name].index
#     index_names_3 = dd1.Temp.index
#     index_names_4 = dd2.Temp.index
#     final_index_name = index_names_1.union(index_names_2).union(index_names_3).union(index_names_4)
#     df.drop(final_index_name, inplace=True)
    
    x = df.loc[:, ['Temp','cdd','hdd','cdd2','hdd2','Hour','Month','DOW','covid']]
    y = df[data_name]
    
    # split the cleaned data
    X_train, X_val, y_train, y_val = split_data(x,y)
    
    return X_train, X_val, y_train, y_val 


# ## 3. plotplot2: calcuate  cvrmse_val, r2_val

# In[6]:


def plotplot2(method, X_train, X_val, y_train, y_val):
    
    # set up stats calculation
    r2_train = r2_score(y_train, method.predict(X_train))
    r2_val = r2_score(y_val, method.predict(X_val))
    rmse_train = math.sqrt(mean_squared_error(y_train, method.predict(X_train)))
    rmse_val = math.sqrt(mean_squared_error(y_val, method.predict(X_val)))
    mae_train = np.median((method.predict(X_train) - y_train.values))
    mae_val = np.median(method.predict(X_val) - y_val.values)
    cvrmse_train = rmse_train/(np.max(method.predict(X_train))-np.min(method.predict(X_train))/2)
    cvrmse_val = rmse_val/(np.max(method.predict(X_val))-np.min((method.predict(X_val))/2))
    return cvrmse_val, r2_val


    #plot of Training Data vs Model Prediction
#     plt.figure()
#     training_comparison = pd.DataFrame({"Actual":y_train,
#                                        "Modeled":method.predict(X_train)}, index=y_train.index)

#     training_comparison.sort_index().plot(figsize=(18,3),
#                              title="Training Data vs Model Prediction",
#                              linewidth=1,
#                              color=["#a53860","#00b4d8"])

#     plt.title(f'Training Data vs Model Prediction \n R2:{round(r2_train,3)}\n CVRMSE:{round(cvrmse_train,3)} \n RMSE:{round(rmse_train,3)} MAE:{round(mae_train,3)}',
#         fontsize = 18)
#     plt.legend(prop={'size': 14},loc='upper left')


#     #plot of Validation Data vs Model Prediction
#     plt.figure()
#     val_comparison = pd.DataFrame({"Actual":y_val,
#                                        "Modeled":method.predict(X_val)}, index=y_val.index)
    
#     ## set then negative values to 0 only in Validation set.
#     val_comparison.loc[val_comparison.iloc[:,1]<0] = 0
    
#     val_comparison.sort_index().plot(figsize=(18,3),
#                          title="Validation Data vs Model Prediction",
#                          linewidth=1,
#                          color= ["#a53860","#00b4d8"])

#     plt.title(f'Validation Data vs Model Prediction \n R2:{round(r2_val,3)}\n CVRMSE:{round(cvrmse_train,3)} \n RMSE:{round(rmse_val,3)} MAE:{round(mae_val,3)}',
#         fontsize = 18)
#     plt.legend(prop={'size': 14},loc='upper left')    

# ## 4. auto_run

# 4. auto_run:
#  - PI
#  - Gradient Boosting
#  - GridSearch
#  - cvrmse_val, r2_val testing
#  - prediction
## data = df or csv file of all tags for 1 year
# In[7]:


def auto_run(data_name, pred_start=None, pred_end=None):

    start = datetime.now() - timedelta(days=365)
    end = datetime.now()
    
    interval = '1h'    
    calc = 'summary'  
    chunk_size = 10

    df = pc.get_stream_by_point(tags, start=start, end=end,
                                _convert_cols='numeric',
                                calculation=calc,interval=interval,
                                _chunk_size=chunk_size)

    # Drop all NaN columns from df
    df2 = df.dropna(axis=1,how='all')
    df2 = df2.ffill()
    data2_name = df2.iloc[:,:-1].columns

    
  
    newdf = pd.DataFrame()
    temperature = pd.DataFrame(df2.aiTIT4045)

    # create a prediction for all tags to put in newdf
    for i in range(len(df2.iloc[:,:-10].columns)):
    #for i in range(15):

        X_train, X_val, y_train, y_val = data_manipu(df2.iloc[:,i].name, df2,
                                                     temperature)
                # scale the data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        gbt = RandomForestRegressor(random_state=42)
        gbt_param_list = {"max_depth":[5,10],
                          "n_estimators":[100,1000]
                         }
                # using gradient boosting method
        gs_gbt = HalvingRandomSearchCV(gbt, gbt_param_list,cv = 3, factor=3,
                                       min_resources='smallest')
        gs_gbt.fit(X_train_scaled, y_train)
                # validation cvrmse > 0.3and r2 <0.6 check, if not, not run therest and put the tag name into the list
        cvrmse_val, r2_val = plotplot2(method = gs_gbt,X_train = X_train_scaled,
                                       X_val = X_val_scaled,
                                       y_train = y_train, y_val= y_val)
                # print() 
                # print(round(cvrmse_val,2), round(r2_val,2))
                # print()

        start = datetime.now() - timedelta(days=365)
        end = datetime.now()
        x_predict = pc.get_stream_by_point('aiTIT4045',  start = start,
                                           end = end, calculation=calc,
                                           interval=interval,
                                           _chunk_size=chunk_size)
        x_predict.rename(columns = {'aiTIT4045':'Temp'}, inplace=True)
        x_predict['cdd'] = 0
        x_predict.loc[x_predict.Temp > 65, 'cdd'] = x_predict.loc[x_predict.Temp > 65, 'Temp'] - 65

        x_predict['hdd'] = 0
        x_predict.loc[x_predict.Temp < 65, 'hdd'] = 65 - x_predict.loc[x_predict.Temp < 65, 'Temp']
        x_predict['cdd2'] = x_predict.cdd**2
        x_predict['hdd2'] = x_predict.hdd**2
        x_predict["Hour"] = x_predict.index.hour
        x_predict["Month"] = x_predict.index.month
        x_predict["DOW"] = x_predict.index.dayofweek
        x_predict['covid'] = 0
                # x_predict.loc["2020-03-17":"2021-09-28", 'covid'] = 1
        x_predict = x_predict.dropna()
        x_predict_scaled = scaler.transform(x_predict)

            # final check
            # checking cvrmse and r2
        if cvrmse_val > 0.5:
            pred = np.nan
            newdf.insert(i, data2_name[i], pred)
            #print(i+1, "out of", len(df2.iloc[:,:-10].columns))
            #print(pd.DataFrame(pred, index=x_predict.index, columns = [data_name[i]]))

        else:
            # put new predicted values in newdf
            pred = gs_gbt.predict(x_predict_scaled)
            newdf.insert(i, data2_name[i], pred)
            
            #print(i+1, "out of", len(df2.iloc[:,:-10].columns))
            #print(pd.DataFrame(pred, index=x_predict.index, columns = [data_name[i]]))
        return newdf

# # Run the script

# In[8]:


#pred_start = "2019-07-05"
#pred_end = "2020-07-05"
#data_name = "Tupper_Electricity_Demand_kBtu"
#data_name ='CHE.Office/Lab_Electricity_Demand_kBtu'


# In[9]:


#c = auto_run(data_name = "Tupper_Electricity_Demand_kBtu", pred_start=pred_start, pred_end=pred_end)


# In[10]:


#c = auto_run(data_name = data_name, pred_start=pred_start, pred_end=pred_end)


# In[11]:


#c.head()


# %%
