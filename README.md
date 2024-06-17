# Breast-cancer
Breast cancer prediction in machine learning involves using algorithms to analyze clinical and histological data to accurately classify whether a tumor is benign or malignant.

 Machine learning is widely used in bio informatics and particularly in breast cancer diagnosis. In this project, we have used certain classification methods such as K-nearest neighbors (K-NN) and Support Vector Machine (SVM) which is a supervised learning method to detect breast cancer. Cancer diagnosis is one of the most studied problems in the medical domain. Several researchers have focused in order to improve performance and achieved to obtain satisfactory results. Early detection of cancer is essential for a rapid response and better chances of cure. Unfortunately, early detection of cancer is often difﬁcult because the symptoms of the disease at the beginning are absent. Thus, it is necessary to discover and interpret new knowledge to prevent and minimize the risk adverse consequences.

To understand this problem more precisely, tools are needed to help oncologists to choose the treatment required for healing or prevention of recurrence by reducing the harmful effects of certain treatments and their costs. In artiﬁcial intelligent, machine learning is a discipline which allows the machine to evolve through a process. Wisconsin Diagnostic Breast Cancer (WDBC) dataset obtained by the university of Wisconsin Hospital is used to classify tumors as benign or malignant.

## EXPLORATORY DATA ANALYSIS
### Understanding the data with libraries
```
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
```
# Reading data into dataframe
```
cancer = pd.read_csv('cancer.csv')
```
# PLOTTING SCATTER MATRIX
```
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
```
![concave](https://github.com/adepel80/Breast-cancer/assets/123180341/e18e464d-32f1-4937-944b-8e0715616a13)

# CHOOSING THE X AND Y VALUE 
```
#Choosing x and y values

#x is our features except diagnosis (classification columns)
#y is diagnosis
x_cancer = cancer.iloc[:,1:]
y = cancer['diagnosis']
```
![breast x y value](https://github.com/adepel80/Breast-cancer/assets/123180341/b164af94-3178-4f2a-89b1-285a8d54915a)

# TRAIN AND TEST
```

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.3, random_state = 1)
print('x-train shape : ', x_train.shape)
print('y-train shape : ', y_train.shape)
print('x-test shape : ', x_test.shape)
print('y-test shape : ', y_test.shape)

```
![breast train and test](https://github.com/adepel80/Breast-cancer/assets/123180341/68850f14-1fb8-4fd5-94d3-3cd680cfeade)

# KNN 
```
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
predcted_value = knn.predict (x_test)
corrected_value = np.array(y_test)

print('kNN(with k=3) accuracy is : ', knn.score(x_test, y_test))
```
![breast knn](https://github.com/adepel80/Breast-cancer/assets/123180341/14011fe6-822f-4c71-8489-c8ca4a1bba9b)
# SVC
```
from sklearn.svm import SVC
svm = SVC(random_state = 1, gamma = 'auto')
svm.fit(x_train, y_train)
print('accuracy of SVM : ', svm.score(x_test, y_test))

```
![breast svc](https://github.com/adepel80/Breast-cancer/assets/123180341/67d9f226-2414-4e78-8337-b6f4e8638665)
# GaussianNB
```
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
print ('accuracy of naive bayes : ', nb.score(x_test, y_test))

```
![breast gaussan](https://github.com/adepel80/Breast-cancer/assets/123180341/dbe9855a-acf1-4a72-b05f-a3945f39e032)
# DECISSION TREE CLASSIFIER
```
# Descision tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
print('accuracy of Decision Tree Classification: ', dt.score(x_test, y_test))
```
![breast decission](https://github.com/adepel80/Breast-cancer/assets/123180341/60375379-14c0-421e-bde3-434be0b9286e)

# RANDOM FOREST CLASSIFIER
```
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(x_train, y_train)
print('accuracy of Random Forest Classification: ', rf.score(x_test, y_test))
```
![breast RF](https://github.com/adepel80/Breast-cancer/assets/123180341/e9bef14f-5011-482c-9189-07d1557e4e32)


# FINDING THE n VALUE FOR Knn

```
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
```
![BREAST KNN visual](https://github.com/adepel80/Breast-cancer/assets/123180341/7d1bdeda-9b07-4aa8-bdac-804760035ae9)

# RESULT 
SVMs shows the accuracy of :  0.631578947368421
kNN(with k=3) accuracy is :  0.9239766081871345
Naive bayes : accuracy of naive bayes :  0.9473684210526315
accuracy of Decision Tree Classification:  0.935672514619883
accuracy of Random Forest Classification:  0.9532163742690059



#k- Nearest Neighbour (k-NN) classification technique:

k-NN is a non- parametric method used for classification. In this classification, the output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor. It is the simplest algorithm among all the machine learning algorithms.

#Support Vector Machine (SVM) classification Technique:

Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for both classification or regression challenges. However, it is mostly used in classification problems. In this algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiate the two classes very well


