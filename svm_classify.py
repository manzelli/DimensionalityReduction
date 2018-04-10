from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals import joblib
from sklearn.svm import SVC




def svm_classify(features, labels):
	train_feat, test_feat, train_lbl, test_lbl = cross_validation.train_test_split(features, labels, test_size=0.2)

	param = [
		{
			"kernel": ["linear"],
			"C"     : [1, 10, 100, 1000]
		},
		{
			"kernel": ["rbf"],
			"C"     : [1, 10, 100, 1000],
			"gamma" : [1e-2, 1e-3, 1e-4, 1e-5]
		}
	]

	# Turn off probability estimation, set decision function to One Versus One
	svm = SVC(probability=False, decision_function_shape='ovo')

	# 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
	clf = grid_search.GridSearchCV(svm, param, cv=10, n_jobs=4, verbose=3)
	clf.fit(train_feat, train_lbl)

	print("\nBest parameters set:")
	print(clf.best_params_)

	# Testing on classifier..
	y_predict = clf.predict(test_feat)

	labels_sort = sorted(list(set(labels)))
	print("\nConfusion matrix:")
	print("Labels: {0}\n".format(",".join(labels_sort)))
	print(confusion_matrix(test_lbl, y_predict, labels=labels_sort))

	print("\nClassification report:")
	print(classification_report(test_lbl, y_predict))