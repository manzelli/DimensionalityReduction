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
from svm_classify_kpca import * 
from sklearn.datasets import fetch_mldata

import math

mnist = fetch_mldata("MNIST original")
xtrain = mnist.data / 255.0
ytrain = mnist.target


n_components = 16
time_start = time.time()
for x in range (0,10):
    kpca = KernelPCA(n_components = n_components, kernel = 'rbf',fit_inverse_transform = True ,gamma = x).fit(xtrain)
    time_end = time.time()
    print("done in %0.3fs" % (time.time() - time_start))
    pca.explained_variance_ratio_.sum() 
    xtrain_pca = kpca.transform(xtrain)
    print(xtrain_pca.shape)
    svm_classify(xtrain_pca,ytrain)




