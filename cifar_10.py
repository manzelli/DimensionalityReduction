import pickle
import sys
import os
import glob
import numpy as np

cifar_path = '/Users/sshong/Desktop/EC503_Project/DimensionalityReduction/cifar_10'
train_names = 'data_batch'
test_name = 'test_batch'
meta = 'batches.meta'

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def get_cifar():
	data_filenames = glob.glob(cifar_path + '/' + train_names + '*')
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
