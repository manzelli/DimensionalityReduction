import pickle
import sys
import os
import glob
import numpy as np

cifar_path = '/home/ubuntu/cifar_10'
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
	labels = first[b'labels']
	filenames = first[b'filenames']
	label_names = (unpickle(cifar_path + '/' + meta))[b'label_names']
	data_filenames.pop(0)
	for file in data_filenames:
		file_dict = unpickle(file)
		data = np.vstack((data, file_dict[b'data']))
		labels = np.vstack((labels, file_dict[b'labels']))
		filenames = np.vstack((filenames, file_dict[b'filenames']))
	return data, labels, filenames, label_names
