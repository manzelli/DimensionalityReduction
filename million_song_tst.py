import os
import sys
import time
import datetime
import glob
import numpy as np

# Location of the Actual Dataset on the mounted EC2 drive
msd_path ='/mnt/snap'
msd_data_path = os.path.join(msd_path, 'data')
msd_addf_path = os.path.join(msd_path, 'AdditionalFiles')

# Location of helper code for accessing the Dataset
msd_code_path='/home/ubuntu/MSongsDB'
assert os.path.isdir(msd_code_path), 'wrong path'
sys.path.append(os.path.join(msd_code_path, 'PythonSrc'))
import hdf5_getters as GETTERS

