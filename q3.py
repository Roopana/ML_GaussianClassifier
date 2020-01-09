#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:59:57 2019

@author: roopana
"""
from sklearn import datasets
from crossValidation import mycrossval
import pandas as pd
import numpy as np
#import warnings
#warnings.filterwarnings('ignore')

def loadDataSets():        
    boston = datasets.load_boston()
    
    ## Creating Boston50 dataset
    dfBoston50 = pd.DataFrame(data=boston['data'], columns = boston['feature_names'])
    dfBoston50['response'] = boston['target']
    dfBoston50['target'] = dfBoston50['response']>= np.percentile(dfBoston50['response'],50)
    dfBoston50['target'] = dfBoston50['target']*1
    
    # Creating Boston75 dataset
    dfBoston75 = pd.DataFrame(data=boston['data'], columns = boston['feature_names'])
    dfBoston75['response'] = boston['target']
    dfBoston75['target'] = dfBoston75['response']>= np.percentile(dfBoston75['response'],75)
    dfBoston75['target'] = dfBoston75['target']*1
    
    #Construct dfDigits Data
    digits = datasets.load_digits(n_class = 10, return_X_y = False)
    dfDigits = pd.DataFrame(data = digits.data)
    dfDigits['target'] = digits.target
    
    return dfBoston50, dfBoston75, dfDigits

# calls cross validationa and thus executing classification
def run():
    dfBoston50, dfBoston75, dfDigits = loadDataSets()

    x50 = dfBoston50.iloc[:, 0:13]
    y50 = dfBoston50.iloc[:, 14] # Take the target column as y and not the actualresponse column
    x75 = dfBoston75.iloc[:, 0:13]
    y75 = dfBoston75.iloc[:, 14]
    xDigits = dfDigits.iloc[:, 0:64]
    yDigits = dfDigits.iloc[:, 64]
    
    k=5 # No of folds for Cross Validation
    nClasses =2 # no of classes in the input dataset
    method = 'multigaussclassify'
    
    print("Applying multigaussclassify on Boston 50 Data")
    mycrossval(method, x50, y50, k, nClasses)
    
    print("Applying multigaussclassify on Boston 75 Data")
    mycrossval(method, x75, y75, k, nClasses)
    
    print("Applying multigaussclassify on Digits Data")
    nClasses = 10
    mycrossval(method, xDigits, yDigits, k, nClasses)
    
    method = 'multigaussdiagclassify'
    nClasses =2
   
    print("Applying multigaussdiagclassify on Boston 50 Data")
    mycrossval(method, x50, y50, k, nClasses)
    
    print("Applying multigaussdiagclassify on Boston 75 Data")
    mycrossval(method, x75, y75, k, nClasses)
   
    print("Applying multigaussdiagclassify on Digits Data")
    nClasses = 10
    mycrossval(method, xDigits, yDigits, k, nClasses)
   
    method = 'LogisticRegression'
    nClasses = 2
   
    print("Applying LogisticRegression on Boston 50 Data")
    mycrossval(method, x50, y50, k, nClasses)
    
    print("Applying LogisticRegression on Boston 75 Data")
    mycrossval(method, x75, y75, k, nClasses)
 
    print("Applying LogisticRegression on Digits Data")
    nClasses = 10
    mycrossval(method, xDigits, yDigits, k, nClasses)

run()