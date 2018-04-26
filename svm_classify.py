from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals import joblib
import numpy as np
from sklearn.svm import SVC


def svm_classify(features, labels, printout=True):
	train_feat, test_feat, train_lbl, test_lbl = cross_validation.train_test_split(features, labels, test_size=0.2)

	c_vals = [10^ element for element in [-3, -2, -1, 0, 1, 2, 3, 4, 5]]
	g_vals = [10^ element for element in [-6, -5, -4, -3, -2, -1, 0, 1, 2]]

	params= [
		{
			"kernel": ["linear"],
			"C"     : c_vals
		},
		{
			"kernel": ["rbf"],
			"C"     : c_vals,
			"gamma" : g_vals
		}
	]

	# Turn off probability estimation, set decision function to One Versus One
	svm = SVC(probability=False, decision_function_shape='ovo')

	# 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
	clf = grid_search.GridSearchCV(svm, params, cv=10, n_jobs=4, verbose=3)
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
