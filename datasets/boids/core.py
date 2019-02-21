import numpy as np
import os

from . import cfg


######################################################################
######################### Data Manipulation ##########################
######################################################################


def fetch(train):
    filename = cfg.FILENAME_TRAIN if train else cfg.FILENAME_TEST
    fullpath = os.path.join(cfg.DATAPATH, filename)

    # Load data
    assert os.path.isfile("{}.npz".format(fullpath))
    data = np.load("{}.npz".format(fullpath))['data']

    # Load macro-intents
    macro_path = os.path.join(cfg.DATAPATH, "macro_intents", "{}_friendly.npz".format(filename))
    assert os.path.isfile(macro_path)
    labels = np.load(macro_path)['data']
    
    assert labels.shape[0] == data.shape[0]

    return data, labels

def preprocess(data, labels):
    return data, labels

def normalize(data):
    return data

def unnormalize(data):
    return data
