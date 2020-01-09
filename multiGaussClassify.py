#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:39:14 2019

@author: vcroopana
"""
import numpy as np
import math

class MultiGaussClassify:
    nClasses = 0
    priorProbability = []
    μ = []
    Σ = []
    ## consructor param k - no of classes in the classification
    def __init__(self, k):
        
        self.nClasses = k
        self.priorProbability = np.zeros(self.nClasses)
        # μ,Σ cannot be initialzed as per the question since the dimension d of data is not known yet.
        # As discussed in class by the prof, leaving it uninitialized since we are not using the value anyways.
        self.μ = [] 
        self.Σ = []
        for i in range(self.nClasses):
            self.priorProbability[i] = 1/self.nClasses 
        
            
    def calculateCovariance(self, XClass, μ, ithClass, diag):
        cov = np.zeros([XClass.shape[1], XClass.shape[1]])
            
        for a in range(XClass.shape[0]):
            rowVector = np.subtract(XClass[a],  μ[ithClass])
            colVector = rowVector.T
            product = np.dot(colVector, rowVector)
            cov = cov + product

        cov = np.divide(cov, XClass.shape[0])
        
        if(diag == True):
            cov = np.multiply(cov, np.identity(cov.shape[0]))
            
        return cov

    # X - feature matrix
    # y - column vector with 'target' column
    # The method updates priorProbability, μ, Σ values as per given traning data
    def fit(self,X,y,diag):
        """
        classes = y.groupby('target',as_index = True, sort = True) ## or y.target.unique() either ways dependant on column name
        namesOfClasses = list(classes.groups.keys()) 
        # no of classes in the given data
        nGroupsInData = len(classes.groups)
        """
        namesOfClasses = sorted(y.unique())
        # no of classes in the given data
        nGroupsInData = len(namesOfClasses)
        
        if self.nClasses != nGroupsInData:
            print('# classes in the data is not equal to no of classes requested as output. This is an inconsistency?')
     
        for i in range(nGroupsInData): # i =0, 1..nGroupsInData
            nameOfIthClass = namesOfClasses[i]
            nSamplesAllClasses = X.shape[0]

            #Rows corresponding to current class
            XClass = X[y == nameOfIthClass]

            # Calculate prior probability of ith class
            self.priorProbability[i] = XClass.shape[0]/ nSamplesAllClasses

            # Calculate mean of ith class
            # Result is a row vector with mean value of each column
            self.μ.append(np.array([XClass.mean(axis = 0)]))
            self.Σ.append( self.calculateCovariance(XClass.values, self.μ, i, diag))
        
    # Code for calculating Conditional Probability
    # input - x - row vector i.e a sample data vector; cov & μ of a class
    # output - log(P(x|Ci))
    def calculateCondProb(self, x, cov, μ, nClasses):
        
        diff =np.subtract(x.values, μ)
        diffT = diff.T
        detCov = np.linalg.det(cov)
        # e value to make cov non singular matrix as given in the question
        e = 0.001
        while detCov == 0:
            e = np.identity(cov.shape[0])*e
            cov = np.add(cov, e)
            detCov = np.linalg.det(cov)
            e = e/10

        covInverse = np.linalg.inv(cov)
        matProd1 = np.dot(diff, covInverse)
        matProduct = np.dot(matProd1,diffT) #1x15, 15x1
        constant = -0.5*nClasses*math.log(2*math.pi)- 0.5*math.log(detCov)
        return (np.add(constant, np.multiply(-0.5,matProduct)))

    # input : X - test set
    # outout: predicted class of X
    def predict(self,X):
        predictedY = np.zeros([X.shape[0]])
        for row in range(X.shape[0]):
            predictedClass = np.zeros([self.nClasses])
            maxY = -float('inf')
            for i in range(self.nClasses):
                logCondProb = self.calculateCondProb(X.iloc[[row]], self.Σ[i], self.μ[i], self.nClasses)
                predictedClass[i] = logCondProb + math.log(self.priorProbability[i])
                if(predictedClass[i]> maxY):
                    maxY = predictedClass[i]
                    predictedY[row] = i

        return predictedY  
            
            
            