import os
import sys
import time
import glob
import datetime
from sklearn import *
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def svm_classify(features, labels, printout=True):
	train_feat, test_feat, train_lbl, test_lbl = model_selection.train_test_split(features, labels, test_size=0.2)

	g_vals = [10^ element for element in [-6, -5, -4, -3, -2, -1, 0, 1, 2]]

	best_params = {
		"kernel": ["rbf"],
		"C"		: [10],
		"gamma" : [10]
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

	classifier = svm.SVC(probability=False, decision_function_shape='ovo', cache_size=72940)

	# 10-fold cross validation, use 4 thread as each fold and each parameter set can train in parallel
	clf = model_selection.GridSearchCV(classifier, params, cv=5, n_jobs=36, verbose=3)
	clf.fit(train_feat, train_lbl)

	# Testing on classifier..
	y_predict = clf.predict(test_feat)

	if printout:
		print("\nBest parameters set:")
		print(clf.best_params_)

		labels_sort = sorted(list(set(labels)))
		print("\nConfusion matrix:")
		print("Labels: {0}\n".format(", ".join(labels_sort)))
		print(confusion_matrix(test_lbl, y_predict, labels=labels_sort))

		print("\nClassification report (per label):")
		print(classification_report(test_lbl, y_predict))

	return clf, y_predict
