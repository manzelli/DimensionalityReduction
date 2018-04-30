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

mnisttrain = pd.read_csv('../mnistdata/train.csv')
xtrain = mnisttrain.drop(['label'], axis='columns', inplace=False)
ytrain = mnisttrain['label']

n_components = 16
time_start = time.time()
kpca = KernelPCA(n_components = n_components, fit_inverse_transform = True, kernel = 'rbf', gamma = gamma, n_jobs = -1).fit(xtrain)
time_end = time.time()
print("done in %0.3fs" % (time.time() - time_start))
x_kpca = kpca.transform(xtrain); 
print(x_kpca.shape)
svm_classify(x_kpca,ytrain)




