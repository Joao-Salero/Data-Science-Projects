#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'>Monte Carlo Simulation</font>
# # <font color='blue'>Monte Carlo Simulation and Time Series for Financial Modeling</font>
# 
# 

# ### Loading the Packages

# In[1]:


# Python Version
from platform import python_version
print('Python Version:', python_version())

# Imports for data manipulation
import numpy as np
import pandas as pd

# Imports for viewing
import matplotlib.pyplot as plt
import matplotlib as m
import seaborn as sns

# Imports for statistical calculations
import scipy
from scipy.stats import kurtosis, skew, shapiro
import warnings
warnings.filterwarnings("ignore")

# Imports for formatting graphics
plt.style.use('fivethirtyeight')
m.rcParams['axes.labelsize'] = 14
m.rcParams['xtick.labelsize'] = 12
m.rcParams['ytick.labelsize'] = 12
m.rcParams['text.color'] = 'k'
from matplotlib.pylab import rcParams 
rcParams['figure.figsize'] = 20,10


# ### Loading Data

# In[2]:


df = pd.read_csv("YOUR_PATH", parse_dates = True, index_col = "Date")


# In[3]:


# View the first lines
df.head()


# In[4]:


# Data Types
df.dtypes


# In[5]:


# Shape
df.shape


# In[6]:


# Summary
df.describe()


# ## Viewing the Daily Share Closing Price

# In[7]:


# Plot
plt.plot(df["Close"])
plt.title("Daily Share Closing Price", size = 14)
plt.show()


# In[8]:


#Calculating the percentage change in the daily closing quote of the shares
daily_return = df["Close"].pct_change().dropna()
daily_return.head()


# In[9]:


# Daily Return
accumulated_daily_return = (1 + daily_return).cumprod() - 1
accumulated_daily_return.max()


# ### Exploratory Analysis and Descriptive Statistics

# Calculation of Average Return and Variation.

# In[10]:


# Daily closing average 
av_return_daily = np.mean(daily_return)


# In[11]:


# Standard Deviation of Daily Closed
dev_daily_return = np.std(daily_return)


# In[12]:


# Mean and Standard Deviation
print("Average Closing Return:", av_return_daily)
print("Standard Deviation of Closing Return:", dev_daily_return)


# Note: Considering 252 Days of Trading on the United States Stock Exchange.

# In[13]:


# Mean and Standard Deviation Per Year 
print("Yearly Average Closing Return:", (1 + av_return_daily) ** 252 - 1)
print("Yearly Standard Deviation of Closing:", dev_daily_return*np.sqrt(252))


# Although the stock's performance has been good in recent years, the average gain is low, but positive. Thus, the investor has not lost money.

# In[14]:


# Plot
plt.plot(daily_return)
plt.title("Daily Return", size = 14)
plt.show()


# The daily return has been constant over time, with only two major variations.

# In[15]:


# Plot
plt.hist(daily_return, bins = 75)
plt.title("Daily Return Histogram", size = 14)
plt.show()


# ### Kurtosis and Skewness

# In[16]:


print("Kurtosis:", kurtosis(daily_return))
print("Skewness:", skew(daily_return))


# Although Kurtosis indicates that the records are close to the mean, Skewness demonstrates a distortion and a non-Normal Distribution of the data.

# ### Shapiro-Wilk Test

# In[17]:


# Run the normality test for the series
is_normal_test_01 = shapiro(daily_return)[1]

# Check return based on p-value of 0.05
if is_normal_test_01 <= 0.05:
    print("Rejects the Null Hypothesis of Data Normality.")
else:
    print("Failure to reject the Null Hypothesis of Data Normality.")


# There is no Normal Distribution.
# 
# To calculate the Daily Return Amount:
# Log transformation to the series and then apply the differencing technique to remove the trend patterns and leave only the real data.

# In[18]:


# Log Transformation and Differentiation
log_daily_return = (np.log(df["Close"]) - np.log(df["Close"]).shift(-1)).dropna()

# Mean and Standard Deviation After Transformation
log_av_daily_return = np.mean(log_daily_return)
log_dev_daily_return = np.std(log_daily_return)


# In[19]:


# Plot
plt.plot(log_daily_return)
plt.title("Daily Return (Log Transformation)", size = 14)
plt.show()


# In[20]:


# Plot
plt.hist(log_daily_return, bins = 75)
plt.title("Daily Return Histogram (Log Transformation)", size = 14)
plt.show()


# In[21]:


# Kurtosis and Skewness
print("Kurtosis:", kurtosis(log_daily_return))
print("Skewness:", skew(log_daily_return))


# In[22]:


# Normality Test for the Series
is_normal_test_02 = shapiro(log_daily_return)[1]

# Return Based on p-Value of 0.05
if is_normal_test_02 <= 0.05:
    print("Rejects the Null Hypothesis of Data Normality.")
else:
    print("Failure to reject the Null Hypothesis of Data Normality.")


# Note: The data is still not normal, despite the reduction of data distortion. There is room for other transformations.

# ### Historical value

# In[23]:


# Variance Level
var_level = 95
var = np.percentile(log_daily_return, 100 - var_level)
print("Assurance that daily losses will not exceed VaR%.")
print("VaR 95%:", var)


# In[24]:


# Var for the next 5 days
var * np.sqrt(5)


# ### Conditional Historical Value

# In[25]:


# Variance Level
var_level = 95
var = np.percentile(log_daily_return, 100 - var_level)
cvar = log_daily_return[log_daily_return < var].mean()
print("In the worst 5% of cases on average losses were higher than the historical percentage.")
print("CVaR 95%:", cvar)


# ### Monte Carlo Simulation

# In[26]:


# Number of Days Ahead
ahead_days = 252

# Number of Simulations
sim = 2500

# Last Share Value
last_price = 270.3

# Empy Array with the dimensions 
res = np.empty((sim, ahead_days))

# Loop por cada simulação
for s in range(sim):
    
    # Calculates the return with random data following a Normal Distribution
    random_returns = 1 + np.random.normal(loc = log_av_daily_return, 
                                          scale = log_dev_daily_return, 
                                          size = ahead_days)
    
    result = last_price * (random_returns.cumprod())
    
    res[s, :] = result


# In[27]:


# Defining the Simulated Series Index
index = pd.date_range("2020-03-11", periods = ahead_days, freq = "D")
results_all = pd.DataFrame(result.T, index = index)
average_results = results_all.apply("mean", axis = 1)


# ## Monte Carlo Simulation Result

# In[28]:


fig, ax = plt.subplots(nrows = 2, ncols = 1)

# Plot
ax[0].plot(df["Close"][:"2018-12-31"])

ax[0].plot(results_all)

ax[0].axhline(270.30, c = "orange")

ax[0].set_title(f"Monte Carlo {sim} Simulation", size = 14)

ax[0].legend(["Historical Price", "Last Price = 270.30"])

ax[1].plot(df["Close"][:"2018-12-31"])

ax[1].plot(results_all.apply("mean", axis = 1), lw = 2)

ax[1].plot(average_results.apply((lambda x: x * (1+1.96 * log_dev_daily_return))), 
           lw = 2, linestyle = "dotted", c = "gray")

ax[1].plot(average_results, lw = 2, c = "orange")

ax[1].plot(average_results.apply((lambda x: x * (1-1.96 * log_dev_daily_return))), 
           lw = 2, linestyle = "dotted", c = "gray")

ax[1].set_title(f"Average Result Monte Carlo {sim} Simulation", size = 14)

ax[1].legend(["Price", "Average Forecast", "2x Standard Deviation"])

plt.show()


# - Positive Forecast.
# - Stocks tend to appreciate in the long term.
# - Do not expect expressive returns.
