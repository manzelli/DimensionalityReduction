import time
import numpy as np
import sklearn as skl
from multiprocessing import cpu_count
from keras.datasets import mnist
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.externals import joblib
from svm_classify import *
from cifar_10 import *
import matplotlib
matplotlib.use('TkAgg')
from ggplot import*

# Following Medium's Visualising high-dimensional datasets using PCA and t-SNE in Python
# https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
def get_mnist():
	# Getting data from MNIST dataset using
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train / 255.0
	x_test = x_test / 255.0
	x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
	x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

	x = np.concatenate((x_train, x_test))
	y = np.concatenate((y_train, y_test))
	return x, y

def my_tsne(x, y=None, components=2, verbose=1, n_iter=500, perp = 30):
	time_start = time.time()
	tsne_red = TSNE(n_components=components, verbose=verbose, n_iter=n_iter, n_jobs=36, perplexity=perp)
	tsne_results = tsne_red.fit_transform(X=x)

	print('t-SNE done! Time running: {} seconds.'.format(time.time() - time_start))
	return tsne_results


def visualise_results(df_tsne):
	chart = ggplot(df_tsne, aes(x=df_tsne['xtsne'], y=df_tsne['ytsne'], color='label')) + geom_point(size=70, alpha=0.1) \
			+ ggtitle("t-SNE dimensions colored by digit")
	chart.show()


def main():
	# Getting data from MNIST dataset
	data, labels = get_mnist()
	print("Performing t-sne on MNIST Dataset... \n")

	# Performing t-SNE dimensionality reduction & graphing reduced dimensions with their original labels
	data_reduced = my_tsne(data, labels, components=2, verbose=1, n_iter=1000, perp=34)

	# Using a support vector machine to classify reduced dimensionality data
	print("Classifing reduced MNIST Data with SVM... \n ")
	svm_classify(data_reduced, labels, printout=True)

	# Getting data from cifar_10 dataset
	data, labels, filenames, label_names = get_cifar()
	data = preprocessing.normalize(data)
	print("Performing t-sne on cifar-10 Dataset... \n")
	data_reduced = my_tsne(data, labels, components=2, verbose=1, n_iter=1000, perp=55)

	print("Classifing reduced cifar_10 Data with SVM... \n ")
	svm_classify(data_reduced, labels, printout=True)


if __name__ == '__main__':
	main()
