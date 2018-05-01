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
import math

cifar_path = '/Users/sshong/Desktop/EC503_Project/DimensionalityReduction/cifar_10'
train_names = 'data_batch_1'
test_name = 'test_batch'
meta = 'batches.meta'

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def get_cifar():
	data_filenames = glob.glob(cifar_path + '/' + train_names)
	data_filenames.append(cifar_path + '/' + test_name)
	first = unpickle(data_filenames[0])
	data = first[b'data']
	labels = np.array(first[b'labels'])
	filenames = np.array(first[b'filenames'])
	label_names = np.array((unpickle(cifar_path + '/' + meta))[b'label_names'])
	data_filenames.pop(0)
	for file in data_filenames:
		file_dict = unpickle(file)
		data = np.vstack((data, file_dict[b'data']))
		labels = np.append(labels, file_dict[b'labels'], 0)
		filenames = np.append(filenames, file_dict[b'filenames'], 0)
	return data, labels, filenames, label_names

def svm_classify(features, labels, printout=True):
	train_feat, test_feat, train_lbl, test_lbl = model_selection.train_test_split(features, labels, test_size=0.2)

	g_vals = [10^ element for element in [-6, -5, -4, -3, -2, -1, 0, 1, 2]]

	best_params = [
                {
                        "kernel":["rbf"],
                        "C"     :[10],
                        "gamma" :[10]
                        }
                ]
	
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
	clf = model_selection.GridSearchCV(classifier, params, cv=2, n_jobs=36, verbose=3)
	clf.fit(train_feat, train_lbl)

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
        xtrain,labels,filenames,label_names = get_cifar()
        ytrain = labels

        n_components = 16
        time_start = time.time()
        kpca = KernelPCA(n_components = n_components, kernel = 'rbf')

        import pickle
        # obj0, obj1, obj2 are created here...
        xtrain_kpca = kpca.fit_transform(xtrain);
        time_end = time.time()
        print("done in %0.3fs" % (time.time() - time_start))
        # Saving the objects:
        with open('xtrain_kpca.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([xtrain_kpca], f)

        print(xtrain_kpca.shape)
        svm_classify(xtrain_kpca,ytrain)

if __name__ == '__main__':
        main()



