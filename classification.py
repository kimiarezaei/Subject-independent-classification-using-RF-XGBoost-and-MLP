# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:49:41 2022

@author: KimiaR
"""


import numpy as np
import pandas as pd
import glob 
from xgboost import XGBClassifier ,XGBRFClassifier    
from sklearn.ensemble import RandomForestClassifier   
import time


start = time.time()             #calculate the algorithm runtime


#read signals and annotations
list_of_features = glob.glob('pdatabase_test/featurespdatabase/*.csv')               # create the list of signals
list_of_labels = glob.glob('pdatabase_test/labelspdatabase/*.csv')               # create the list of signals

#reading features or annotations from CSV files
def my_data(list):   
    X_total = np.array([])
    for no in range(len(list)):
        X=pd.read_csv(list[no])
        X=X.dropna()
        X=np.array(X)
        X_total=np.append(X_total,X)
        X_total=X_total.reshape(-1,1) 
    return X_total

#classifier and leave one out method for evaluation   

#applying leave one out method on list of files
g = ([[lf], [el for el in list_of_features if el is not lf]] for lf in list_of_features)
g2 = ([[lf], [el for el in list_of_labels if el is not lf]] for lf in list_of_labels)
for x,y in zip(g,g2): 
                                       
    #all files but not one for training
    X_train=my_data(x[1])
    y_train=my_data(y[1])
    #only one file each iterartion for testing
    X_test=my_data(x[0])
    y_test=my_data(y[0])  




    #random forest for Classification
    
    clf1 = RandomForestClassifier(max_depth=2, random_state=40)      #using random forest for classification
    clf1.fit(X_train, np.ravel(y_train))              #fit the model --np ravel does some reshaping(for removing some errors)   
    
    
    
    
    #XGboost for Classification
    clf2= XGBClassifier()                         #using XGBoost for classification
    clf2.fit(X_train, np.ravel(y_train))              #fit the model --np ravel does some reshaping(for removing some errors)   
    
    
  
    #XGBoost Random Forest for Classification
    clf3 = XGBRFClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.2)       #using XGBoost random forest for classification
    clf3.fit(X_train, np.ravel(y_train))              #fit the model --np ravel does some reshaping(for removing some errors)   
    
    
    
#accuracy 
acc1 = clf1.score(X_test, y_test)
print('Accuracy of random forest classifier: %.9f' % acc1)
acc2 = clf2.score(X_test, y_test)
print('Accuracy of XGboost classifier: %.9f' % acc2)
acc3 = clf3.score(X_test, y_test)
print('Accuracy of XGBoost Random Forest classifier: %.9f' % acc3)



end = time.time()                              
total_time = end - start
print("\n"+ str(total_time))


