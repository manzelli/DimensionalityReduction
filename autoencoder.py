# EC503 Final Project: Exploration of Dimensionality Reduction
# Implementation of Simple Autoencoder in Keras (fully connected)
# REFERENCE: https://blog.keras.io/building-autoencoders-in-keras.html

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # plot digits
from keras.callbacks import TensorBoard # for logging
from sklearn import svm
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals import joblib
from sklearn.svm import SVC
np.set_printoptions(threshold=np.inf)

# size of our encoded representation
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
# Rectified linear activation is our activation function, to prevent vanishing gradient
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input, with standard sigmoid activation
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

### ENCODER ###
# this model maps an input to its encoded representation. This is what we want to explore.
encoder = Model(input_img, encoded)

### DECODER ###
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Configure model using adadelta optimizer for gradients and cross entropy loss function
autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')

### DATA PREPROCESS ###
(x_train, train_labels), (x_test, test_labels) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape) # Encodes from 784 dimensions to 32 dimensions

# We fit the data to itself, since we want to reconstruct input data
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]) # to check loss

# encode and decode some digits from test set
encoded_test = encoder.predict(x_test) # 0-9
encoded_train = encoder.predict(x_train)
print(encoded_train.shape)
print(encoded_test.shape)
decoded_imgs = decoder.predict(encoded_test)

# CLASSIFY 
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
clf.fit(encoded_train, train_labels)

print("\nBest parameters set:")
print(clf.best_params_)

# Testing on classifier..
y_predict = clf.predict(encoded_test)

labels_sort = sorted(list(set(train_labels)))
print("\nConfusion matrix:")
print("Labels: {0}\n".format(",".join(labels_sort)))
print(confusion_matrix(test_labels, y_predict, labels=labels_sort))

print("\nClassification report:")
print(classification_report(test_labels, y_predict))


# Classify
# clf = svm.SVC(decision_function_shape='ovo')
# clf.fit(encoded_data, train_labels)
# decisions = clf.predict(encoded_imgs)
# CCR = (decisions == test_labels) / len(test_labels)
# print(CCR)

# Plots
# n = 10  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

# n = 10
# plt.figure(figsize=(20, 8))
# for i in range(n):
#     ax = plt.subplot(1, n, i + 1)
#     plt.imshow(encoded_imgs[i].reshape(4, 4 * 2).T)
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()