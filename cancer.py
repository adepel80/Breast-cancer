#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,r2_score
plt.style.use ("dark_background")


# In[12]:


pwd


# In[13]:


data = pd.read_csv('data.csv')


# In[14]:


data


# In[15]:


data.info()


# In[16]:


plt.figure(figsize=(10,5))
sns.histplot(data=data['diagnosis'],kde=True,palette='hot')


# In[17]:


nan = data.isna()
nan.sum()


# In[18]:


sns.heatmap(nan)


# In[19]:


data = data.drop('Unnamed: 32',1) # 1 represent column bases, 0 represent row based
data.head()


# In[20]:


data.head()


# In[21]:


data.describe()


# In[22]:


sns.heatmap(nan)


# In[23]:


data.head()


# In[24]:


diagnosis = LabelEncoder()
data['diagnosis'] = diagnosis.fit_transform(data['diagnosis'])
data


# In[25]:


data.corr()


# In[26]:


corr = data.corr()
corr


# In[27]:


plt.figure(figsize=(15, 12))
sns.heatmap(corr)


# In[28]:


plt.figure(figsize=(40, 20))
sns.heatmap(data.corr())
top_corr_features = data.corr().index
g=sns.heatmap(data[top_corr_features].corr(),annot=True,linewidth=.10,cmap="rocket")


# In[29]:


plt.figure(figsize=(40, 20))
matrix = np.triu(data.corr())
sns.heatmap(data.corr(), annot=True, linewidth=.10, mask=matrix, cmap="Paired");


# In[30]:


plt.figure(figsize=(17, 10))
sns.histplot(data=corr,kde=True,palette='hot')


# In[31]:


data.nunique()


# In[32]:


unique = pd.DataFrame(data.nunique())


# In[33]:


plt.figure(figsize=(15, 10))
sns.heatmap(unique,annot=True, linewidth=.10, cmap="Paired")


# In[34]:


sns.countplot (data['diagnosis'])


# In[35]:


data


# # splitting data for train and test data

# In[36]:


y = data.iloc[:,1].values   #lables
y


# In[37]:


x = data.drop('diagnosis',1) # 1 represent column bases, 0 represent row based
x


# In[38]:


x = np.array(x)
x


# In[39]:


SC = StandardScaler()
x[:,0:5] = SC.fit_transform(x[:,0:5])
x[:,10:14] = SC.fit_transform(x[:,10:14])
x


# # apply PCA to reduction dimention

# In[40]:


sns.heatmap(x)


# In[41]:


x_train,x_test,y_train,y_test = train_test_split ( x , y , test_size = 0.2 , random_state = 0)


# In[42]:


x_test


# In[43]:


DF_x_test = pd.DataFrame(x_test,columns = ['id','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'])
DF_x_test


# # applying an algo (KNeighborsClassifier) to make prediction before Normalization

# In[44]:


KNC = KNeighborsClassifier()
KNC.fit(x_train,y_train)
y_pred5 = KNC.predict(x_test)
print(y_pred5)


# In[45]:


cm = confusion_matrix(y_test, y_pred5)
get_ipython().run_line_magic('matplotlib', 'inline')
# Plot confusion matrix
class_names = ['M','B']
df_cm = pd.DataFrame(cm, index = [i for i in class_names], columns = [i for i in class_names])
sns.heatmap(df_cm, annot = True)
cmap = plt.cm.Blues
plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
# Model Accuracy
acc = accuracy_score(y_test, y_pred5)
print("Model Accuracy =",acc*100,"%")


# # applying an algo (SVM) to make prediction before Normalization

# In[46]:


svc = SVC()
svc.fit(x_train,y_train)
y_pred2 = svc.predict(x_test)
print(y_pred2)


# In[47]:


cm = confusion_matrix(y_test, y_pred2)
get_ipython().run_line_magic('matplotlib', 'inline')
# Plot confusion matrix
class_names = ['M','B']
df_cm = pd.DataFrame(cm, index = [i for i in class_names], columns = [i for i in class_names])
sns.heatmap(df_cm, annot = True)
cmap = plt.cm.Blues
plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
# Model Accuracy
acc = accuracy_score(y_test, y_pred2)
print("Model Accuracy =",acc*100,"%")


# # applying an algo (DecisionTreeClassifier) to make prediction before Normalization¶¶

# In[48]:


DTC = DecisionTreeClassifier()
DTC.fit(x_train,y_train)
y_pred3 = DTC.predict(x_test)
print(y_pred3)


# In[49]:


cm = confusion_matrix(y_test, y_pred3)
get_ipython().run_line_magic('matplotlib', 'inline')
# Plot confusion matrix
class_names = ['M','B']
df_cm = pd.DataFrame(cm, index = [i for i in class_names], columns = [i for i in class_names])
sns.heatmap(df_cm, annot = True)
cmap = plt.cm.Blues
plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
# Model Accuracy
acc = accuracy_score(y_test, y_pred3)
print("Model Accuracy =",acc*100,"%")


# # applying an algo (Random Forest) to make prediction before Normalization

# In[50]:


RFC = RandomForestClassifier()
RFC.fit(x_train,y_train)
y_pred4 = RFC.predict(x_test)
print(y_pred4)


# In[51]:


cm = confusion_matrix(y_test, y_pred4)
get_ipython().run_line_magic('matplotlib', 'inline')
# Plot confusion matrix
class_names = ['M','B']
df_cm = pd.DataFrame(cm, index = [i for i in class_names], columns = [i for i in class_names])
sns.heatmap(df_cm, annot = True)
cmap = plt.cm.Blues
plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
# Model Accuracy
acc = accuracy_score(y_test, y_pred4)
print("Model Accuracy =",acc*100,"%")


# In[ ]:




