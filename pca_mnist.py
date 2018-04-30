import numpy as numpy
import pandas as pd 

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib

import time
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from svm_classify_pca import * 
from sklearn.datasets import fetch_mldata

import math

mnisttrain = pd.read_csv('./mnistdata/train.csv')
xtrain = mnisttrain.drop(['label'], axis='columns', inplace=False)
ytrain = mnisttrain['label']
xtrain = xtrain / 255.0

# X_tr, X_ts, Y_tr, Y_ts = train_test_split(xtrain,ytrain,test_size=0.30,random_state=4)

n_components = 16
time_start = time.time()
pca = PCA(n_components = n_components, svd_solver = 'randomized',whiten = True).fit(xtrain)
time_end = time.time()
print("done in %0.3fs" % (time.time() - time_start))
pca.explained_variance_ratio_.sum() 

xtrain_pca = pca.transform(xtrain)
print(xtrain_pca.shape)
# xtrain_kpca = kpca.transform(xtrain)

svm_classify(xtrain_pca,ytrain, printout = True)




