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
from svm_classify import * 
from sklearn.datasets import fetch_mldata

import math

mnist = fetch_mldata("MNIST original")
xtrain = mnist.data / 255.0
ytrain = mnist.target

# X_tr, X_ts, Y_tr, Y_ts = train_test_split(xtrain,ytrain,test_size=0.30,random_state=4)

n_components = 16
time_start = time.time()
pca = PCA(n_components = n_components, svd_solver = 'randomized',whiten = True).fit(xtrain)
#kpca = KernelPCA(n_components = n_components, kernel = 'rbf',fit_inverse_transform = True ,gamma = 10).fit(xtrain)
time_end = time.time()
print("done in %0.3fs" % (time.time() - time_start))
pca.explained_variance_ratio_.sum() 

xtrain_pca = pca.transform(xtrain)
xtrain_kpca = kpca.transform(xtrain)

svm_classify(xtrain_pca,ytrain)
# param = {"C" : [0.1], "gamma":[0.1]}
# rf = SVC()
# # search for the best parameters for PCA data 
# gs_pca = GridSearchCV(estimator=rf, param_grid=param, scoring='accuracy', cv=2, n_jobs=-1, verbose=1)
# gs_pca = gs_pca.fit(xtrain_pca, ytrain)

# # #search for the best parameters for regular data
# # gs = GridSearchCV(estimator=rf, param_grid=param, scoring='accuracy', cv=2, n_jobs=-1, verbose=1)
# # gs = gs.fit(xtrain, ytrain)

# print(gs_pca.best_score_)
# print(gs_pca.best_params_)

# bp_pca = gs_pca.best_fpfarams_
# # bp = gs.best_params_

# # t0 = time()
# # Train SVM with regular data 
# # clf = SVC(C = 0.1,kernel = 'rbf', gamma = 0.1)
# # clf = clf.fit(xtrain,ytrain)
# # print("done training SVM in %0.3fs" % (time() - t0))

# t1 = time()
# # Train SVM with PCA Model 
# clf_pca = SVC(C=bp_pca['C'], kernel='rbf', gamma=bp_pca['gamma'])
# clf_pca = clf_pca.fit(xtrain_pca, ytrain)
# clf_kpca = SVC(C=bp_pca['C'], kernel='rbf', gamma=bp_pca['gamma'])
# clf_kpca = clf_kpca.fit(xtrain_kpca, ytrain)
# print("done training SVM with PCA in %0.3fs" % (time() - t1))
# print(clf_pca.score(pca.transform(X_ts),Y_ts))
# print(clf_kpca.score(kpca.transform(X_ts),Y_ts))



