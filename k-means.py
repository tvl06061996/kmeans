# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:27:46 2019

@author: loc.tran
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KMeans(object):
    def __init__(self, k = 3, epsilon = 0.0001, max_iterations = 500):
        self.k = 3
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        
    def fit(self, X):
        self.centers = {}
        self.classes = {}
        
        # init center
        for i in range(self.k):
            self.centers[i] = X[i]

        for i in range(self.max_iterations):
            for i in range(self.k):
                self.classes[i]= []
                
            for feature in X:
                distances = [np.linalg.norm(feature - self.centers[center]) for center in self.centers]
                nearest = distances.index(min(distances))
                self.classes[nearest].append(feature)
             
            previous = dict(self.centers)
            # update centers
            for center in self.centers:
                self.centers[center] = np.average(self.classes[center], axis=0)
            
            isOptimal = False
            for center in self.centers:
                original_center = previous[center]
                current_center = self.centers[center]
                if sum((current_center - original_center) / original_center) * 100 > self.epsilon:
                    isOptimal = True
            if isOptimal:
                break