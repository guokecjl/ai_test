# coding: utf-8
import numpy as np
from sklearn.neural_network import MLPClassifier as DNN
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score as cv
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import train_test_split as TTS
from time import time
import datetime
data = load_breast_cancer()
X = data.data
y = data.target
Xtrain, Xtest, Ytrain, Ytest = TTS(X,y,test_size=0.2,random_state=420)

dnn = DNN(hidden_layer_sizes=(200,100),max_iter=500,random_state=420)
dnn = dnn.fit(Xtrain, Ytrain)
print(dnn.score(Xtest, Ytest))
