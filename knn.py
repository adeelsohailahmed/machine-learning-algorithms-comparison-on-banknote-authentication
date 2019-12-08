# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:20:17 2019

@author: Adeel Ahmed
"""

import numpy as np
import pandas as pd
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Create a dataframe of the dataset from its CSV file
df = pd.read_csv('data_banknote_authentication.csv')

# Convert the dataframe into Numpy array
df = np.array(df)

# Set a fixed seed to ensure that the order of permutation is always the same
np.random.seed(42)

# Create random permutation from 0 to the total number of rows in the dataset
L = np.random.permutation(len(df))

# Permute the rows according to the order specified in L
permuted_dataset = df[L]

# Slice the dataset into data and target respectively
data = permuted_dataset[:, :4] # First four columns contain features of images
target = permuted_dataset[:, 4] # Last column contains the class of images

# Since the 4 out of 5 columns of this dataset contain float, dataframe converts the
# last column to float automatically. Therefore, convert the target column from float
# to integer again for good measure (although it doesn't affect the prediction).
target = np.array(target, int)

# N is number of rows used to test the model after the training
N = 200    

# Separate training data and testing data along with their respective targets
data_train = data[:-N]
target_train = target[:-N]

data_test = data[-N:]
target_test = target[-N:]

print("\nBinary Classification Using K-Nearest Neighbours Algorithm")
print("============================================================")

# Run the K-Nearest Neighbour Model with different values of K and generate report
for K in [3, 5, 7]:
   
    # Start measuring the time
    time_start = time.perf_counter()
    
    knn = KNeighborsClassifier(n_neighbors = K)    
    knn.fit(data_train, target_train)
    
    target_predicted = knn.predict(data_test)
    
    # After the classes have been predicted, stop measuring the time and note the difference
    time_end = time.perf_counter() - time_start
    
    print('\nFor  K =', K)
    print('----------')
    
    print(classification_report(target_test, target_predicted,
                                target_names=['Genuine Banknotes (0)', 'Forged Banknotes (1)']))
    
    print('\nAccuracy Score:', accuracy_score(target_test, target_predicted))