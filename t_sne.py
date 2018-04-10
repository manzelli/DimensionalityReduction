import time
import numpy as np
import sklearn
from sklearn.datasets import fetch_mldata
from sklearn.manifold import TSNE
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals import joblib
from itertools import combinations
from sklearn.svm import SVC
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ggplot import*

# Following Medium's Visualising high-dimensional datasets using PCA and t-SNE in Python


def t_sne(dataframe, samplefactor=.1, components=2, verbose=1, n_iter=300):
	time_start = time.time()
	numsamples = int(df.shape[0] * samplefactor)
	randperm = np.random.permutation(dataframe.shape[0])
	time_start = time.time()
	tsne = TSNE(n_components=components, verbose=verbose, n_iter=n_iter)
	tsne_results = tsne.fit_transform(dataframe.loc[randperm[:numsamples], feat_cols].values)

	df_tsne = dataframe.loc[randperm[:numsamples], :].copy()
	df_tsne['xtsne'] = tsne_results[:, 0]
	df_tsne['ytsne'] = tsne_results[:, 1]
	print('t-SNE done! Time elapsed: {} seconds.'.format(time.time() - time_start))
	return tsne_results, df_tsne


def visualise_results(df_tsne):
	chart = ggplot(df_tsne, aes(x=df_tsne['xtsne'], y=df_tsne['ytsne'], color='label')) + geom_point(size=70, alpha=0.1) \
			+ ggtitle("t-SNE dimensions colored by digit")
	chart.show()


def svm_classify(features, labels):
	train_feat, test_feat, train_lbl, test_lbl = cross_validation.train_test_split(features,labels,test_size=0.2)

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


# Getting example data from MNIST dataset
mnist = fetch_mldata("MNIST original")
x = mnist.data / 255.0
y = mnist.target
print(x.shape, y.shape)

# Putting data into a Pandas DataFrame
feat_cols = ['pixel' + str(i) for i in range(x.shape[1])]
df = pd.DataFrame(x, columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i))
x = None
print('Size of dataframe: {}'.format(df.shape))

# Performing t-SNE dimensionality reduction & graphing reduced dimensions with their original labels
tsne_results, dataframe_tsne = t_sne(df, samplefactor=.1, components=2, verbose=1, n_iter=500)
#visualise_results(dataframe_tsne)

# Using a support vector machine to classify reduced dimensionality data
svm_classify(tsne_results[:, [0, 1]], y)
