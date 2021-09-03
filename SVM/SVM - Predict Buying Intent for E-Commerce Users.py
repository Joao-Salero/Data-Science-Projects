#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'>Using SVM to Predict Buying Intent for E-Commerce Users</font>
# 

# In[1]:


# Python Version
from platform import python_version
print('Python Version for this Jupyter Notebook:', python_version())


# ## Importing Package
# 
# To update a package, run the command below in the terminal or command prompt:
# pip install -U pack_name
# 
# After to installing or updating de packcage, restart the Jupyter Notebook. 
# 

# In[2]:


import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn import svm
from imblearn.over_sampling import SMOTE
import sklearn
import warnings
warnings.filterwarnings('ignore')


# ## Loading Data
# 

# In[3]:


# Loading Data
df_original = pd.read_csv('/content/online_shoppers_intention.csv')
df_original.head()


# ## Exploratory Analysis

# In[4]:


# Shape
df_original.shape


# In[5]:


# Data Types
df_original.dtypes


# In[6]:


# Checking missing values
print(df_original.isna().sum())


# In[7]:


# Removing missing values lines
df_original.dropna(inplace = True)


# In[8]:


# Checking missing values
print(df_original.isna().sum())


# In[9]:


# Shape
df_original.shape


# In[10]:


# Checking Unique Values
df_original.nunique()


# Obs: For visualization purposes, we will split the data into continuous and categorical variables. We will treat all variables with less than 30 unique entries as categorical.

# In[11]:


# Preparing the data for the plot

# Create a copy of the original dataset
df = df_original.copy()

# Empty lists for results
continuous = []
categorical = []

# Loop through columns
for c in df.columns[:-1]:
    if df.nunique()[c] >= 30:
        continuous.append(c)
    else:
        categorical.append(c)


# In[12]:


continuous


# In[13]:


# Continuous
df[continuous].head()


# In[14]:


categorical


# In[15]:


# Categorical
df[categorical].head()


# ## Graphs for Continuous

# In[16]:


# Plot Area Size
fig = plt.figure(figsize = (12,8))

# Loop through continuous
for i, col in enumerate(continuous):
    plt.subplot(3, 3, i + 1);
    df.boxplot(col);
    plt.tight_layout()
    
#plt.savefig('imagens/boxplot1.png')


# Obs: Continuous variables look extremely distorted. Let's apply log transformation for better visualization.

# In[17]:


# Log in continuous
df[continuous] = np.log1p(1 + df[continuous])


# In[18]:


# Plot continuous with Log

# Plot Area Size
fig = plt.figure(figsize = (12,8))

# Loop through continuous
for i,col in enumerate(continuous):
    plt.subplot(3,3,i+1);
    df.boxplot(col);
    plt.tight_layout()
    
#plt.savefig('imagens/boxplot2.png')


# ## Correlation Matrix Between Continuous Variables.

# In[19]:


# Plot Area Size
plt.figure(figsize = (10,10))

# Correlation Matrix 
sns.heatmap(df[['Administrative_Duration',
                'Informational_Duration',  
                'ProductRelated_Duration',
                'BounceRates', 
                'ExitRates', 
                'PageValues',
                'Revenue']].corr(), vmax = 1., square = True)


# ## Graph visualization of categorical variables to analyze how the target variable is influenced by them
# 

# In[20]:


# Countplot: Sell or Not
plt.subplot(1,2,2)
plt.title("Sell or Not")
sns.countplot(df['Revenue'])


# In[21]:


# Countplot: Visitor Type
plt.xlabel("Visitor Type")
sns.countplot(df['VisitorType'])


# In[22]:


# Stacked Bar: Visitor Type x Revenue
pd.crosstab(df['VisitorType'], df['Revenue']).plot(kind = 'bar', 
                                                   stacked = True, 
                                                   figsize = (15, 5), 
                                                   color = ['red', 'green'])


# In[23]:


# Pie Chart: Visitor Type
labels = ['Returning_Visitor', 'New_Visitor', 'Others']
plt.title("Visitor Type")
plt.pie(df['VisitorType'].value_counts(), labels = labels, autopct = '%.2f%%')
plt.legend()


# In[24]:


# Countplot: Weekend or Not
plt.subplot(1,2,1)
plt.title("Weekend or Not")
sns.countplot(df['Weekend'])


# In[25]:


# Stacked Bar: Weekend x Revenue
pd.crosstab(df['Weekend'], df['Revenue']).plot(kind = 'bar', 
                                               stacked = True, 
                                               figsize = (15, 5), 
                                               color = ['red', 'green'])


# In[26]:


# Countplot: OS Types
plt.figure(figsize = (15,6))
plt.title("Types of Operational System")
plt.xlabel("Used Operacional Syatem")
sns.countplot(df['OperatingSystems'])


# In[27]:


# Stacked Bar: SO Type x Revenue
pd.crosstab(df['OperatingSystems'], df['Revenue']).plot(kind = 'bar', 
                                                        stacked = True, 
                                                        figsize = (15, 5), 
                                                        color = ['red', 'green'])


# In[28]:


# Countplot: Traffic type
plt.title("Traffic Type")
plt.xlabel("Traffic Type")
sns.countplot(df['TrafficType'])


# In[29]:


# Stacked Bar: Traffic type x Revenue
pd.crosstab(df['TrafficType'], df['Revenue']).plot(kind = 'bar', 
                                                   stacked = True, 
                                                   figsize = (15, 5), 
                                                   color = ['red', 'green'])


# ## Pre-Processing

# In[30]:


df_original.head()


# In[31]:


# Encoder
lb = LabelEncoder()

# Apply the encoder to variables that have string
df_original['Month'] = lb.fit_transform(df_original['Month'])
df_original['VisitorType'] = lb.fit_transform(df_original['VisitorType'])

# Removes eventually generated missing values
df_original.dropna(inplace = True)


# In[32]:


df_original.head()


# In[33]:


# Shape
df_original.shape


# In[34]:


# Checking if the response variable is balanced
target_count = df_original.Revenue.value_counts()
target_count


# In[35]:


# Explanatory variables
df_original.iloc[:, 0:17].head()


# In[36]:


# Target
df_original.iloc[:, 17].head()


# In[37]:


# Plot 
sns.countplot(df_original.Revenue, palette = "OrRd")
plt.box(False)
plt.xlabel('Revenue per session  No (0) / Yes (1) \n', fontsize = 11)
plt.ylabel('Total Session', fontsize = 11)
plt.title('Class Count \n')
plt.show()


# ## Oversampling

# In[38]:


# Seed if you want to reproduce the same result
seed = 100

# X e y
X = df_original.iloc[:, 0:17]  
y = df_original.iloc[:, 17] 

# SMOTE
smote = SMOTE(random_state = seed)

# Applying SMOTE
X_res, y_res = smote.fit_resample(X, y)


# In[39]:


# Plot 
sns.countplot(y_res, palette = "OrRd")
plt.box(False)
plt.xlabel('Revenue per session  No (0) / Yes (1) \n', fontsize = 11)
plt.ylabel('Total Session', fontsize = 11)
plt.title('Class Count \n')
plt.show()


# In[40]:


# Original data shape
df_original.shape


# In[41]:


# Shape of resampled data
print('X:',X_res.shape, 'and', 'Y:',y_res.shape)


# In[42]:


# Resampling X e y
X = X_res
y = y_res


# In[43]:


# Division into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# ## Model SVM

# ### Base Model with Linear Kernel

# In[44]:


# Model V1
model_v1 = svm.SVC(kernel = 'linear')


# In[45]:


# Training
start = time.time()
model_v1.fit(X_train, y_train)
end = time.time()
print('Model Training Time:', end - start)


# In[46]:


# Prediction
pred_v1 = model_v1.predict(X_test)


# In[47]:


# Metrics and Metadata Dictionary
SVM_dict_v1 = {'Model':'SVM',
               'Version':'1',
               'Kernel':'Linear',
               'Precision':precision_score(pred_v1, y_test),
               'Recall':recall_score(pred_v1, y_test),
               'F1 Score':f1_score(pred_v1, y_test),
               'Accuracy':accuracy_score(pred_v1, y_test),
               'AUC':roc_auc_score(y_test, pred_v1)}

print("Test Metrics:\n")
SVM_dict_v1


# ### Model with Linear Kernel and Standardized Data (Scaled)
# 

# In[48]:


# Standardization
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# In[49]:


X_train_scaled


# In[50]:


X_test_scaled


# In[51]:


# Model V2
model_v2 = svm.SVC(kernel = 'linear')


# In[52]:


# Training
start = time.time()
model_v2.fit(X_train_scaled, y_train)
end = time.time()
print('Model Training Time:', end - start)


# In[53]:


# Prediction
pred_v2 = model_v2.predict(X_test_scaled)


# In[54]:


# Metrics and Metadata Dictionary
SVM_dict_v2 = {'Model':'SVM',
               'Version':'2',
               'Kernel':'Linear with Standardized Data',
               'Precision':precision_score(pred_v2, y_test),
               'Recall':recall_score(pred_v2, y_test),
               'F1 Score':f1_score(pred_v2, y_test),
               'Accuracy':accuracy_score(pred_v2, y_test),
               'AUC':roc_auc_score(y_test, pred_v2)}

print("Test Metrics:\n")
SVM_dict_v2


# ### Hyperparameter Optimization with Grid Search and RBF Kernel

# In[55]:


# Model V3
model_v3 = svm.SVC(kernel = 'rbf')

# Values for the grid
C_range = np.array([45., 50., 55.,  100., 200., 300.])
gamma_range = np.array([0.001, 0.01, 0.1, 0.15, 1])
d_range2 = np.array([2, 2.5, 4])
r_range2 =  np.array([0.5, 1])

# Hyperparameter Grid
svm_param_grid = dict(gamma = gamma_range, C = C_range, degree = d_range2, coef0 = r_range2)

# Grid Search
start = time.time()
model_v3_grid_search_rbf = GridSearchCV(model_v3, svm_param_grid, cv = 4)

# Training
model_v3_grid_search_rbf.fit(X_train_scaled, y_train)
end = time.time()
print('Model Training Time with Grid Search:', end - start)

# Training Accuracy
print(f"Training Accuracy: {model_v3_grid_search_rbf.best_score_ :.2%}")
print("")
print(f"Ideal Hyperparameters: {model_v3_grid_search_rbf.best_params_}")


# In[56]:


# Prediction
pred_v3 = model_v3_grid_search_rbf.predict(X_test_scaled)


# In[57]:


# Metrics and Metadata Dictionary
SVM_dict_v3 = {'Model':'SVM',
               'Version':'3',
               'Kernel':'RBF com Dados Padronizados',
               'Precision':precision_score(pred_v3, y_test),
               'Recall':recall_score(pred_v3, y_test),
               'F1 Score':f1_score(pred_v3, y_test),
               'Accuracy':accuracy_score(pred_v3, y_test),
               'AUC':roc_auc_score(y_test, pred_v3)}

print("Test Metrics:\n")
SVM_dict_v3


# ### Hyperparameter Optimization with Grid Search and Polynomial Kernel
# 

# In[58]:


# Model V4
model_v4 = svm.SVC(kernel = 'poly')

# Values for the grid
r_range =  np.array([0.5, 1])
gamma_range =  np.array([0.001, 0.01])
d_range = np.array([2, 3, 4])

# Hyperparameter Grid
param_grid_poly = dict(gamma = gamma_range, degree = d_range, coef0 = r_range)

# Grid Search
start = time.time()
model_v4_grid_search_poly = GridSearchCV(model_v4, param_grid_poly, cv = 3)

# Training
model_v4_grid_search_poly.fit(X_train_scaled, y_train)
end = time.time()
print('Model Training Time with Grid Search:', end - start)

# Training Accuracy
print(f"Training Accuracy: {model_v4_grid_search_poly.best_score_ :.2%}")
print("")
print(f"Ideal Hyperparameters: {model_v4_grid_search_poly.best_params_}")


# In[59]:


# Prediction
pred_v4 = model_v4_grid_search_poly.predict(X_test_scaled)


# In[60]:


# Metrics and Metadata Dictionary
SVM_dict_v4 = {'Model':'SVM',
               'Version':'4',
               'Kernel':'Polynomial with Standardized Data',
               'Precision':precision_score(pred_v4, y_test),
               'Recall':recall_score(pred_v4, y_test),
               'F1 Score':f1_score(pred_v4, y_test),
               'Accuracy':accuracy_score(pred_v4, y_test),
               'AUC':roc_auc_score(y_test, pred_v4)}

print("Test Metrics:\n")
SVM_dict_v4


# In[61]:


# Concatenate all dictionaries into a Pandas dataframe
summary = pd.DataFrame({'SVM_dict_v1':pd.Series(SVM_dict_v1),
                       'SVM_dict_v2':pd.Series(SVM_dict_v2),
                       'SVM_dict_v3':pd.Series(SVM_dict_v3),
                       'SVM_dict_v4':pd.Series(SVM_dict_v4)})

summary


# ### Prediction with the Trained Model

# In[67]:


# New registry
new_x = np.array([4.0, 5.56, 1.0, 3.78, 2.995, 6.00, 0.69, 0.70, 0.69, 0, 6, 1, 1, 3, 3, 2, False]).reshape(1, -1)


# In[68]:


# Standardizing the registry
new_x_scaled = StandardScaler().fit_transform(new_x)


# In[69]:


# Prediction
pred_new_x = model_v3_grid_search_rbf.predict(new_x_scaled)


# In[70]:


pred_new_x


# # End
