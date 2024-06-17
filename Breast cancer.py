#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install chart_studio')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
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

import os


# In[3]:


pwd


# In[4]:


cancer = pd.read_csv('cancer.csv')


# In[5]:


cancer.info()


# In[6]:


# Drop the unnecessary columns for the prediction
cancer = cancer.drop(['Unnamed: 32', 'id'], axis=1)


# In[7]:


cancer['diagnosis'].value_counts()


# In[8]:


cancer.info()


# In[9]:


color_list = ['red' if i == 'M' else 'blue' for i in cancer.loc[:,'diagnosis']]
pd.plotting.scatter_matrix(cancer.iloc[:, 7:13],
                                       c=color_list,
                                       figsize= [10,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '.',
                                       edgecolor= "black")
plt.show()


# In[10]:


cancer['diagnosis'] = [1 if x=='M' else 0 for x in cancer['diagnosis']]


# In[11]:


cancer.head()


# In[12]:


corr = cancer.corr()
corr


# In[13]:


#Choosing x and y values

#x is our features except diagnosis (classification columns)
#y is diagnosis
x_cancer = cancer.iloc[:,1:]
y = cancer['diagnosis']


# In[14]:


x = (x_cancer - np.min(x_cancer) / (np.max(x_cancer) - np.min(x_cancer)))


# In[15]:


x.head()


# In[16]:


#train test split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.3, random_state = 1)


# In[17]:


print('x-train shape : ', x_train.shape)
print('y-train shape : ', y_train.shape)
print('x-test shape : ', x_test.shape)
print('y-test shape : ', y_test.shape)


# In[18]:


cancer.head()


# In[19]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
predcted_value = knn.predict (x_test)
corrected_value = np.array(y_test)

print('kNN(with k=3) accuracy is : ', knn.score(x_test, y_test))


# In[20]:


from sklearn.svm import SVC
svm = SVC(random_state = 1, gamma = 'auto')
svm.fit(x_train, y_train)
print('accuracy of SVM : ', svm.score(x_test, y_test))


# In[21]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
print ('accuracy of naive bayes : ', nb.score(x_test, y_test))


# In[23]:


# Descision tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
print('accuracy of Decision Tree Classification: ', dt.score(x_test, y_test))


# In[24]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(x_train, y_train)
print('accuracy of Random Forest Classification: ', rf.score(x_test, y_test))


# In[25]:


# find best n value for knn
best_neig= range(1,25) 
train_accuracy_list =[]
test_accuracy_list =[]

for each in best_neig:
    knn = KNeighborsClassifier(n_neighbors =each)
    knn.fit(x_train,  y_train)
    train_accuracy_list.append( knn.score(x_train, y_train))    
    test_accuracy_list.append( knn.score(x_test, y_test))    
    
        
print( 'best k for Knn : {} , best accuracy : {}'.format(test_accuracy_list.index(np.max(test_accuracy_list))+1, np.max(test_accuracy_list)))
plt.figure(figsize=[13,8])
plt.plot(best_neig, train_accuracy_list,label = 'Train Accuracy')
plt.plot(best_neig, test_accuracy_list,label = 'Test Accuracy')
plt.title('Neighbors vs accuracy ')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.xticks(best_neig)
plt.show()


# 
