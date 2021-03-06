import numpy as np
import pandas as pd 
import os
import sys
import time
import glob
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import time
import random
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.datasets import fetch_mldata

import math

def svm_classify(features, labels, printout=True):
    train_feat, test_feat, train_lbl, test_lbl = train_test_split(features, labels, test_size=0.2)

    g_vals = [10^ element for element in [-6, -5, -4, -3, -2, -1, 0, 1, 2]]

    best_params = {
		"kernel": ["rbf"],
		"C"		: [1],
		"gamma" : [0.1]
    }

    kernel_params = {"kernel": ["rbf"],"C": [.01, .1, 1, 10, 100, 1000, 10000],"gamma" : [.001, .01, .1, 1, 10, 100, 1000, 10000]}

	# Turn off probability estimation, set decision function to One Versus One

    classifier = SVC(probability=False, decision_function_shape='ovo', cache_size=72940)

	# 10-fold cross validation, use 4 thread as each fold and each parameter set can train in parallel
    clf = GridSearchCV(classifier, kernel_params, cv=2, n_jobs=36, verbose=3)
    clf.fit(train_feat, train_lbl)

    scores = [x[1] for x in clf.grid_scores_]
    scores = np.array(scores).reshape(len(kernel_params["C"]),len(kernel_params["gamma"]))

    plt.figure()
    for ind, i in enumerate(kernel_params["C"]):
        plt.plot(np.log10(kernel_params["gamma"]), scores[ind], label = 'C: ' + str(i))
    plt.legend()
    plt.xlabel('Log-scaled Gamma')
    plt.ylabel('Mean Score')
    plt.savefig('./pca_results/gridsearch_rbf_pca_mnist.png')

	# Testing on classifier..
    y_predict = clf.predict(test_feat)

    if printout:
        print("\nBest parameters set:")
        print(clf.best_params_)

        labels_sort = sorted(list(set(labels)))
        print("\nConfusion matrix:")
        print("Labels: {0}\n".format(", ".join(str(labels_sort))))
        print(confusion_matrix(test_lbl, y_predict, labels=labels_sort))

        print("\nClassification report (per label):")
        print(classification_report(test_lbl, y_predict))

    return clf, y_predict

def main():

    mnisttrain = fetch_mldata('MNIST original')
    xtrain = mnisttrain.data
    ytrain = mnisttrain.target

    #reduce to 16 dimensions
    n_components = 16
    time_start = time.time()
    # obstain the first 16 eigenvectors and fit them 
    pca = PCA(n_components = n_components, svd_solver = 'randomized',whiten = True).fit(xtrain)
    time_end = time.time()
    print("done in %0.3fs" % (time.time() - time_start))
    xtrain_pca = pca.transform(xtrain)
    xtrain_inv_proj = pca.inverse_transform(xtrain_pca)

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        index = random.randint(1,60000)
        print(index)
        ax = plt.subplot(2, n, i + 1)
        img = xtrain[index]
        plt.imshow(np.reshape(img,(28,28)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        #display decoded img 
        ax = plt.subplot(2, n, i + 1 + n)
        img = xtrain_inv_proj[index]
        plt.imshow(np.reshape(img,(28,28)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('./pca_results/pca_mnist.png')

    #classify using the reduced data 
    svm_classify(xtrain_pca,ytrain, printout = True)


if __name__ == '__main__':
    main()

