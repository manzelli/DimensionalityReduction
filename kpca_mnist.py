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
xtrain = xtrain / 255.0

n_components = 16
time_start = time.time()
#xtrain, xtest, ytrain, ytest= model_selection.train_test_split(xtrain, ytrain, test_size=0.)
kpca = KernelPCA(n_components = n_components, kernel = 'rbf')
time_end = time.time()
print("done in %0.3fs" % (time.time() - time_start))

import pickle
# obj0, obj1, obj2 are created here...
x_kpca = kpca.fit_transform(xtrain);

# Saving the objects:
with open('x_kpca.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([x_kpca], f)

    
print(x_kpca.shape)
svm_classify(x_kpca,ytrain)


