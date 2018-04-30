import numpy as numpy
import pandas as pd 

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib

import time
from svm_classify_kpca import *
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import fetch_mldata
from sklearn import model_selection

import math

mnisttrain = pd.read_csv('./mnistdata/train.csv')
xtrain = mnisttrain.drop(['label'], axis='columns', inplace=False)
ytrain = mnisttrain['label']

n_components = 16
time_start = time.time()
#xtrain, xtest, ytrain, ytest= model_selection.train_test_split(xtrain, ytrain, test_size=0.8)
kpca = KernelPCA(n_components = n_components, kernel = 'rbf',fit_inverse_transform=True,gamma=10)
time_end = time.time()
print("done in %0.3fs" % (time.time() - time_start))
x_kpca = kpca.fit_transform(xtrain); 
print(x_kpca.shape)
svm_classify(x_kpca,ytrain)




