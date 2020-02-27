# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:30:38 2019

@author: krish.naik
"""

# Artificial Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sklearn
import torch
import keras
from sklearn.metrics import roc_auc_score

# Importing the dataset
dataset = pd.read_csv('part2_training_data.csv')
X_train = dataset.iloc[:, :dataset.shape[1]-2]
y_train = dataset.iloc[:, dataset.shape[1]-1:]


dataset_val = pd.read_csv('part2_validation.csv')
X_test = dataset_val.iloc[:, :dataset_val.shape[1]-2]
y_test = dataset_val.iloc[:, dataset_val.shape[1]-1:]

# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# print("X train:")
# print(X_train)
# print("Y train")
# print(y_train)

# print("Shape:")
# print(y_train.shape)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Perform Hyperparameter Optimization

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu, sigmoid


def auc_roc_score(y_true, y_pred):
    print(y_true)
    print(y_pred)
    print(type(y_true))
    print(type(y_pred))
    # print("roc-auc score", sklearn.metrics.roc_auc_score(y_true, y_pred))
    # time.sleep(5)
    return sklearn.metrics.roc_auc_score(y_true, y_pred)

METRICS = [
      keras.metrics.Accuracy(name='accuracy'),
      keras.metrics.AUC(name='auc'),
]

def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units = 1, kernel_initializer= 'glorot_uniform', activation = 'sigmoid')) # Note: no activation beyond this point
    
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=METRICS)
    return model
    
model = KerasClassifier(build_fn=create_model, verbose=0)


layers = [[1], [2, 2], [2, 3, 5]]
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)

print("Next shape")
print(y_train.shape)
grid_result = grid.fit(X_train, y_train)

print(grid_result.best_score_,grid_result.best_params_)

pred_y = grid.predict(X_test)
y_pred = (pred_y > 0.5)


from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_pred,y_test)

score = accuracy_score(y_pred,y_test)

print("Confusion Matrix:")
print(cm)

print("Score:")
print(score)













