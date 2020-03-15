# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:21:22 2020

@author: Ishan
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score

data=pd.read_csv(r'C:\Users\Ishan\Documents\Python Scripts\Datasets\Diabetes - KNN.csv')
print(data.head())
no_zeros=["Pregnancies","BloodPressure","SkinThickness","BMI","Insulin"]
for column in no_zeros:
    data[column]=data[column].replace(0,np.NaN)
    mean=int(data[column].mean(skipna=True))
    data[column]=data[column].replace(np.NaN,mean)

X=data.iloc[:,0:8]
y=data.iloc[:,8]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

knn=KNeighborsClassifier(n_neighbors=11,p=2,metric="euclidean")
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(f1_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))