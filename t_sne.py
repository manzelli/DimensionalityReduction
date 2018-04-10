import time
import numpy as np
import sklearn
from sklearn.datasets import fetch_mldata
from sklearn.manifold import TSNE
from svm_classify import svm_classify
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
tsne_results, dataframe_tsne = t_sne(df, samplefactor=.1, components=2, verbose=1, n_iter=250)
#visualise_results(dataframe_tsne)
x1 = tsne_results[:, 0]
x2 = tsne_results[:, 1]
labels = [x1, x2]
# Using a support vector machine to classify reduced dimensionality data
svm_classify(labels, y)
