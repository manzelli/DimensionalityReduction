import numpy as np
import pandas as pd 

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib

import time
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import fetch_mldata
from sklearn import model_selection

import math

def plot_number(row, w=28, h=28, labels=True):
    if labels:
        # the first column contains the label
        try:
            label = row['labels']
        except:
            label = row[0]
        # The rest of columns are pixels
        pixels = row[1:]
    else:
        label = ''
        # The rest of columns are pixels
        pixels = row[0:]
        

    # Make those columns into a array of 8-bits pixels
    # This array will be of 1D with length 784
    # The pixel intensity values are integers from 0 to 255
    pixels = 255-np.array(pixels, dtype='uint8')

    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((w, h))

    # Plot
    if labels:
        plt.title('Label is {label}'.format(label=label))
    plt.imshow(pixels, cmap='gray')


def plot_slice(rows, size_w=28, size_h=28, labels=True):
    num = rows.shape[0]
    w = 4
    h = math.ceil(num / w)
    fig, plots = plt.subplots(h, w)
    fig.tight_layout()

    for n in range(0, num):
        s = plt.subplot(h, w, n+1)
        s.set_xticks(())
        s.set_yticks(())
        plot_number(rows.ix[n], size_w, size_h, labels)

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

	classifier = SVC(probability=False, decision_function_shape='ovo', cache_size=72940)

	# 10-fold cross validation, use 4 thread as each fold and each parameter set can train in parallel
	clf = model_selection.GridSearchCV(classifier, best_params, cv=2, n_jobs=36, verbose=3)
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
    mnisttrain = pd.read_csv('./mnistdata/train.csv')
    xtrain = mnisttrain.drop(['label'], axis='columns', inplace=False)
    ytrain = mnisttrain['label']
    xtrain = xtrain / 255.0

    n_components = 16

    time_start = time.time()
    kpca = KernelPCA(n_components = n_components, kernel = 'rbf')
    xtrain_kpca = kpca.fit_transform(xtrain)
    time_end = time.time()
    print("done in %0.3fs" % (time.time() - time_start))

    train = pd.DataFrame(xtrain)
    reducedtrain = pd.DataFrame(xtrain_kpca)
    reducedtrain = reducedtrain.set_index(ytrain.index)
    reducedtrain.insert(loc = 0, column = 'labels', value = ytrain)
    reducedtrain.reset_index(drop=True, inplace = True)
    plot_slice(pd.DataFrame(data=mnisttrain[0:12]), size_w=28, size_h=28, labels=True)
    plt.savefig('./kpca_results/mnist.png')
    plot_slice(pd.DataFrame(data=reducedtrain[0:12]), size_w=4, size_h=4, labels=True)
    plt.savefig('./kpca_results/kpca_mnist.png')

    import pickle
    # obj0, obj1, obj2 are created here...

    # Saving the objects:
    with open('x_kpca.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([xtrain_kpca], f)

    svm_classify(xtrain_kpca,ytrain)

if __name__ == '__main__':
    main()

