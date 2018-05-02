import numpy as numpy
import pandas as pd 

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import os
import sys
import time
import glob
import datetime

from sklearn import *
from sklearn.metrics import confusion_matrix, classification_report
import time
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from cifar_10 import *

import math

def svm_classify(features, labels, printout=True):
	train_feat, test_feat, train_lbl, test_lbl = model_selection.train_test_split(features, labels, test_size=0.2)

	g_vals = [10^ element for element in [-6, -5, -4, -3, -2, -1, 0, 1, 2]]

	best_params = {
		"kernel": ["rbf"],
		"C"		: [1],
		"gamma" : [0.1]
	}

	params = [
		{
			"kernel": ["linear"],
			"C"     : [.001, .01, .1, 1, 10, 100, 1000, 10000]
		},
		{
			"kernel": ["rbf"],
			"C"     : [.01, .1, 1, 10, 100, 1000, 10000],
			"gamma" : [.001, .01, .1, 1, 10, 100, 1000, 10000]
		}
	]

	# Turn off probability estimation, set decision function to One Versus One

	classifier = SVC(probability=False, decision_function_shape='ovo', cache_size=72940)

	# 10-fold cross validation, use 4 thread as each fold and each parameter set can train in parallel
	clf = model_selection.GridSearchCV(classifier, params, cv=2, n_jobs=36, verbose=3)
	clf.fit(train_feat, train_lbl)

	scores = [x[1] for x in clf.grid_scores_]
	scores = np.array(scores).reshape(len(params[0]["C"]))
	
	for ind in enumerate(params[0]["C"]):
                plt.plot(params[0]["C"], scores[ind])
	plt.xlabel('C')
	plt.ylabel('Mean Score')
	plt.savefig('./pca_results/GridSearch_pca_cifar_10.png')
	
	scores = [x[1] for x in clf.grid_scores_]
	scores = np.array(scores).reshape(len(params[1]["C"]),len(params[1]["gamma"]))

	for ind, i in enumerate(params[1]["C"]):
		plt.plot(params[1]["Gamma"], scores[ind], label = 'C: ' + str(i))
	plt.legend()
	plt.xlabel('Gamma')
	plt.ylabel('Mean Score')
	plt.savefig('./pca_results/GridSearch_rbf_pca_cifar_10.png')
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
    xtrain,ytrain,filenames,label_names = get_cifar()
    print(label_names)

    n_components = 16
    time_start = time.time()
    pca = PCA(n_components = n_components, svd_solver = 'randomized',whiten = True).fit(xtrain)
    time_end = time.time()
    print("done in %0.3fs" % (time.time() - time_start))
    pca.explained_variance_ratio_.sum()
    
    xtrain_pca = pca.transform(xtrain)
    train = pd.DataFrame(xtrain)

    svm_classify(xtrain_pca,ytrain)

if __name__ == '__main__':
    main()




