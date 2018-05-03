from svm_classify import *
from cifar_10 import *
import matplotlib
matplotlib.use('TkAgg')


data, labels = get_cifar()
most_feat, data_subset, most_lbl, label_subset = model_selection.train_test_split(data, labels, test_size=.1)
svm_classify(data_subset, label_subset, printout=True)