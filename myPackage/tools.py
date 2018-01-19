from os.path import isfile, join, altsep
from os import listdir, makedirs, errno
from natsort import natsorted, ns
import numpy as np

def makeDir(path):
    '''
    To create output path if doesn't exist
    see: https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
    :param path: path to be created
    :return: none
    '''
    try:
        makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def natSort(list):
    '''
    Sort frames with human method
    see: https://pypi.python.org/pypi/natsort
    :param list: list that will be sorted
    :return: sorted list
    '''
    return natsorted(list, alg=ns.IGNORECASE)

def getFiles(path):
    samples = [altsep.join((path, f)) for f in listdir(path)
              if isfile(join(path, f)) and f.endswith('.txt')]
    return samples

def splitFiles(array, substr_R, substr_L):
    right_array = [name for name in array if substr_R in name]
    left_array = [name for name in array if substr_L in name]
    other_array = [name for name in array if name not in right_array+left_array]
    return right_array, left_array, other_array

def getData(array):
    data_right = np.empty((0,2), dtype="i8")
    data_left = np.empty((0,2), dtype="i8")
    data_other = np.empty((0,2), dtype="i8")
    for i in range(len(array)):
        for file in array[i]:
            if i == 0:
                # print("Right")
                # data_right.append(np.genfromtxt(file, delimiter = ',', usecols= (-2, -1), dtype="i8"))
                data_right = np.concatenate([data_right, np.genfromtxt(file, delimiter = ',', usecols= (-2, -1), dtype="i8")])
            elif i == 1:
                # print("Left")
                # data_left.append(np.genfromtxt(file, delimiter =',', usecols=(-2, -1), dtype="i8"))
                data_left = np.concatenate([data_left, np.genfromtxt(file, delimiter=',', usecols=(-2, -1), dtype="i8")])
            elif i == 2:
                # print("Other")
                # data_other.append(np.genfromtxt(file, delimiter =',', usecols=(-2, -1), dtype="i8"))
                data_other = np.concatenate([data_other, np.genfromtxt(file, delimiter=',', usecols=(-2, -1), dtype="i8")])
    return data_right, data_left, data_other