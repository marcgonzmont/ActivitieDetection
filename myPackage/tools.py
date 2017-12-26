from os.path import isfile, join, altsep, sep
from os import listdir, makedirs, errno, rename
from natsort import natsorted, ns
import glob
import fnmatch

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

def getSamples(path, endsWith):
    samples = [altsep.join((path, f)) for f in listdir(path)
              if glob(join(f,endsWith))]  #f.endswith(endsWith) and
    return samples

def getSamples2(path, endsWith):
    samples = [f for f in glob.glob(join(path, endsWith))]
    return samples