# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 20:30:29 2023

@author: Rahul
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel("D:\\DS\\books\\ASSIGNMENTS\\Forecasting\\CocaCola_Sales_Rawdata.xlsx")
df
df.shape
df.info()
df.describe()

df.isnull().any()
# There are no null values

#we will extract year and quarter values from Quarter column
df['Quarters'] = 0
df['Year'] = 0
for i in range(42):
    p = df["Quarter"][i]
    df['Quarters'][i]= p[0:2]
    df['Year'][i]= p[3:5]
    
df

# Getting Dummies
Quarters_Dum = pd.DataFrame(pd.get_dummies(df['Quarters']))
df = pd.concat([df,Quarters_Dum],axis=1)
df

import matplotlib.pyplot as plt
import seaborn as sns

#boxplot of Quarters Vs. Sales
sns.set(rc={'figure.figsize':(8,5)})
sns.boxplot(x="Quarters",y="Sales",data=df)

# boxplot of Years Vs. Sales
sns.boxplot(x="Year",y="Sales",data=df)

#density plot
ax =plt.axes()
ax.set_facecolor('black')
df["Sales"].plot(kind='kde',figsize=(8,5),color='blue')

# Lineplot for Sales of CocaCola
plt.figure(figsize=(8,6))
ax = plt.axes()
ax.set_facecolor("white")
plt.plot(df['Sales'], color = 'black', linewidth=3)
plt.xlabel('Year')
plt.ylabel("number of passengers")
plt.show()

df["Sales"].hist()

# Lagplot
from pandas.plotting import lag_plot
plt.figure(figsize=(8,5))
lag_plot(df['Sales'])
plt.show()

plt.figure(figsize=(12,4))
df.Sales.plot(label="org")
for i in range(2,8,2):
    df["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')

plt.figure(figsize=(8,5))
sns.lineplot(x='Year',y='Sales',data=df) #lineplot for year and sales

#heatmap
plt.figure(figsize=(12, 7))
heatmap_y_month = pd.pivot_table(data=df,values="Sales",index="Year",columns="Quarters",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(df.Sales,lags=12)
tsa_plots.plot_pacf(df.Sales,lags=12)
plt.show()

#Timeseries decomposition plot
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_ts_add = seasonal_decompose(df.Sales,period=12)
decompose_ts_add.plot()
plt.show()


#forecasting models
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
  
# Splitting data into Train and Test (77/33)
Train = df.head(32)
Test = df.tail(10)

def RMSE(org, pred):
    rmse=np.sqrt(np.mean((np.array(org)-np.array(pred))**2))
    return rmse
    

#Simple Exponential Method
ses = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses.predict(start = Test.index[0],end = Test.index[-1])
rmse_ses_model = RMSE(Test.Sales, pred_ses)
rmse_ses_model

#Holt method
hw = Holt(Train["Sales"]).fit()
pred_hw = hw.predict(start = Test.index[0],end = Test.index[-1])
rmse_hw_model = RMSE(Test.Sales, pred_hw)
rmse_hw_model

#Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_ad_ad = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=4).fit()
pred_hwe_add_add = hwe_model_ad_ad.predict(start = Test.index[0],end = Test.index[-1])
rmse_hwe_add_add_model = RMSE(Test.Sales, pred_hwe_add_add)
rmse_hwe_add_add_model

#Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=4).fit() 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
rmse_hwe_model_mul_add_model = RMSE(Test.Sales, pred_hwe_mul_add)
rmse_hwe_model_mul_add_model

#Model based Forecasting Methods
# Data preprocessing for models
df["t"] = np.arange(1,43)
df["t_squared"] = df["t"]*df["t"]

df["log_sales"] = np.log(df["Sales"])

df.head()

# Splitting data into Train and Test (77/33)
Train = df.head(32)
Test = df.tail(10)
Train.head()

#Linear Model
import statsmodels.formula.api as smf 
linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear_model = RMSE(Test['Sales'], pred_linear)
rmse_linear_model

#Exponential Model
Exp = smf.ols('log_sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp_model = RMSE(Test['Sales'], np.exp(pred_Exp))
rmse_Exp_model

#Quadratic Model
Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad_model = RMSE(Test['Sales'], pred_Quad)
rmse_Quad_model


#Additive Seasonality model
add_sea = smf.ols('Sales~Q1+Q2+Q3',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1', 'Q2', 'Q3']]))
rmse_add_sea = RMSE(Test['Sales'], pred_add_sea)
rmse_add_sea

#Additive Seasonality Quadratic model
add_sea_Quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','t','t_squared']]))
rmse_add_sea_Quad_model = RMSE(Test['Sales'], pred_add_sea_quad)
rmse_add_sea_Quad_model

#Multiplicative Seasonality model
Mul_sea = smf.ols('log_sales~Q1+Q2+Q3',data=Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mul_sea = RMSE(Test['Sales'], np.exp(pred_Mult_sea))
rmse_Mul_sea

#Multiplicative Additive Seasonality model
Mul_Add_sea = smf.ols('log_sales~t+Q1+Q2+Q3',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mul_Add_sea = RMSE(Test['Sales'], np.exp(pred_Mult_add_sea))
rmse_Mul_Add_sea


list = [['Simple Exponential Method',rmse_ses_model], ['Holt method',rmse_hw_model],
          ['HW exp smoothing add',rmse_hwe_add_add_model],['HW exp smoothing mult',rmse_hwe_model_mul_add_model],
          ['Linear Mode',rmse_linear_model],['Exp model',rmse_Exp_model],['Quad model',rmse_Quad_model],
          ['add seasonality',rmse_add_sea],['Quad add seasonality',rmse_add_sea_Quad_model],
          ['Mult Seasonality',rmse_Mul_sea],['Mult add seasonality',rmse_Mul_Add_sea]]

df1 = pd.DataFrame(list, columns =['Model', 'RMSE_Value']) 
df1

#so quadratic add seasonality model is preferred among them

#Building final model with least RMSE value

final_model = smf.ols('Sales~t+t_squared+Q1+Q2+Q3',data=df).fit()
pred_final = pd.Series(final_model.predict(df[['Q1','Q2','Q3','t','t_squared']]))
rmse_final_model = RMSE(df['Sales'], pred_final)
rmse_final_model

# Actual Vs Predicted graph
sns.set_style('whitegrid')

plt.rcParams['figure.figsize'] = (10, 6) 
_, ax = plt.subplots()
ax.hist(df.Sales, color = 'm', alpha = 0.5, label = 'actual', bins=7)
ax.hist(pred_final, color = 'c', alpha = 0.5, label = 'prediction', bins=7)
ax.yaxis.set_ticks(np.arange(0,11))
ax.legend(loc = 'best')
plt.show()


# Plot of Actual Sales values and Predicted sales values
plt.plot(df.Sales, color='b',marker='o', label='Actual Sales of CocaCola')
plt.plot(pred_final, color='m',marker='x', label='Predicted Sales of CocaCola')

# Added titles and adjust dimensions
plt.title('Actual Sales values and Predicted sales')
plt.xlabel("Timeline")
plt.ylabel("Sales")
plt.legend()
plt.rcParams['figure.figsize'] = (10,8) 
plt.show()

#quad add seasonality has the least mean squarred error among other models
#so quadratic add seasonality model is preferred among them























































    
    
    