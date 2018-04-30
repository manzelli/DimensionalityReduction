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
from cifar_10 import *

import math

xtrain,labels,filenames,label_names = get_cifar()

n_components = 16
time_start = time.time()
kpca = KernelPCA(n_components = n_components, kernel = 'rbf')

import pickle
# obj0, obj1, obj2 are created here...
x_kpca = kpca.fit_transform(xtrain);
time_end = time.time()
print("done in %0.3fs" % (time.time() - time_start))
# Saving the objects:
with open('x_kpca.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([x_kpca], f)

print(x_kpca.shape)
svm_classify(x_kpca,ytrain)




