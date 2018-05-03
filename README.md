# Dimensinality Reduction for the Purpose of Classification

We investigate the effects of dimensionality reduction methods on multi-class classification of image data. Using four methods, namely principal component analysis, kernel principal component analysis, t-SNE, and fully-connected autoencoders, we explore the image data of reduced feature dimension output from each method. We classify this image content with one-vs-one multi-class SVMs, and evaluate performance using correct classification rate, precision, recall, and F1-score. We use the results of the multi-class SVM classification of the original data as a baseline method for comparison of evaluation of each method under investigation.

## Getting Started

To have our code run successfully one needs a python 3.4+ environment with libraries & datasets downloaded & described as rub below

### Installing

A python environment needs to be setup with the following (or similar commands depending on the system)

Intstall python 3.4+
```
sudo apt-get update
sudo apt-get install python3.6
sudo apt-get install python3-pip
```

Necessary Python Libraries:
```
pip3 install tensorflow
pip3 install keras
pip3 install matplotlib
pip3 install sklearn
pip3 install numpy
```


Obtaining the Datasets
```
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
Mnist dataset found at: http://yann.lecun.com/exdb/mnist/
```

## Running the tests

* t-SNE: run the t_sne.py file
* Autoencoder: run the autoencoder.py file
* PCA: run the pca_mnist.py and  pca_cifar_10.py
* k-PCA: run the kpca_mnist.py and  kpca_cifar_10.py

* To get the SVM's results on the original data run original_classify.py

## Authors

* **Nicholas Arnold** - *t-sne implementation, setup on AWS-EC2 instance, SVM* - https://github.com/nickarnold97
* **Rachel Manzelli** - *autoencoder implementation, data plotting* - https://github.com/manzelli
* **Soon Sung Hong** -  *PCA and k-PCA implementations, selection of metrics* - https://github.com/sshong19

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.
