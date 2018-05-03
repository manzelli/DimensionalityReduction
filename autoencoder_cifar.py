# EC503 Final Project: Exploration of Dimensionality Reduction
# Implementation of Simple Autoencoder in Keras (fully connected)
# REFERENCE: https://blog.keras.io/building-autoencoders-in-keras.html
# Followed tutorial and added SVM classifier from:
# http://scikit-learn.org/stable/modules/svm.html

# FOR CIFAR

from sklearn import model_selection
from cifar_10 import *
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt # plot digits
from keras.callbacks import TensorBoard # for logging
from sklearn import svm
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn import preprocessing

np.set_printoptions(threshold=np.inf)

encoding_dim = 32  # dimension we want to encode

input_img = Input(shape=(3072,)) # for cifar size of input is 3072 features
# Rectified linear activation is our activation function, to prevent vanishing gradient
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input, with standard sigmoid activation
decoded = Dense(3072, activation='sigmoid')(encoded)

# maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

### ENCODER ###
# this model maps an input to its encoded representation. This is what we want to explore.
encoder = Model(input_img, encoded)

### DECODER ###
encoded_input = Input(shape=(encoding_dim,))
# last layer of the autoencoder
decoder_layer = autoencoder.layers[-1]
# create decoder
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Configure model using rmsprop optimizer for gradients and mse loss function
autoencoder.compile(optimizer='rmsprop', loss='mean_squared_error')

### DATA PREPROCESS ###
# Get cifar10 from Nick's function
[data, labels, otherthing, names] = get_cifar()
x_train, x_test, train_labels, test_labels = model_selection.train_test_split(data, labels, test_size=0.2)

# Normalize data and reshape 
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape) # Encodes from 3072 dimensions to 32 dimensions

# We fit the data to itself, since we want to reconstruct input data
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]) # to check loss

# encode and decode digits from test set
encoded_test = encoder.predict(x_test) # 0-9
encoded_train = encoder.predict(x_train)

# normalize each feature to increase probability of convergence  of svm :)
encoded_train = preprocessing.normalize(encoded_train)
encoded_test = preprocessing.normalize(encoded_test)

print(encoded_train.shape)
print(encoded_test.shape)

decoded_imgs = decoder.predict(encoded_test)

# CLASSIFY (FROM SKLEARN)
# grid search parameters - box constraint & gamma
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
svm = SVC(probability=False, decision_function_shape='ovo', cache_size=71680)

# 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
# so it doesn't perform really slowly
clf = grid_search.GridSearchCV(svm, param, cv=10, n_jobs=4, verbose=3)
clf.fit(encoded_train, train_labels)

print("\nBest parameters set:")
print(clf.best_params_)

# Testing on classifier..
y_predict = clf.predict(encoded_test)

# labels_sort = ','.join(map(str, train_labels))
# labels_sort = sorted(list(set(labels_sort)))

labels_sort = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] # these are placeholders for the real labels
print("\nConfusion matrix:")
print("Labels: {0}\n".format(",".join(labels_sort)))
print(confusion_matrix(test_labels, y_predict, labels=labels_sort))

print("\nClassification report:")
print(classification_report(test_labels, y_predict))

# Plots
n = 10  # how many digits we display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    img = x_test[i][0:1024] # get only one channel of encoded img
    plt.imshow(img.reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    img = decoded_imgs[i][0:1024]
    plt.imshow(img.reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(encoded_test[i].reshape(4, 4 * 2).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
