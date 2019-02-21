import numpy as np
import os

from . import cfg
from .label_macro_intents import label_macro_intents


######################################################################
######################### Data Manipulation ##########################
######################################################################


def fetch(train, window_size=0):
    filename = cfg.FILENAME_TRAIN if train else cfg.FILENAME_TEST
    fullpath = os.path.join(cfg.DATAPATH, filename)

    # Load data
    assert os.path.isfile("{}.npz".format(fullpath))
    data = np.load("{}.npz".format(fullpath))['data']

    N = cfg.N_TRAIN if train else cfg.N_TEST
    assert data.shape == (N, cfg.SEQUENCE_LENGTH, cfg.SEQUENCE_DIMENSION)

    # Get path to macro-intents
    macro_path = os.path.join(cfg.DATAPATH, "macro_intents", "{}_macro".format(filename))
    if window_size > 0:
        macro_path += "_window{}.npz".format(window_size)
    else:
        macro_path += "_stationary.npz".format(window_size)

    # Compute macro-intents if not saved
    if not os.path.isfile(macro_path):
        label_macro_intents(window_size=window_size)

    # Load macro-intents
    labels = np.load(macro_path)['data']
    assert labels.shape[0] == data.shape[0]

    return data, labels


def preprocess(data, labels):
    # Get just the offensive players
    inds_data = cfg.COORDS['offense']['xy']
    inds_labels = [int(i/2) for i in inds_data[::2]]
    data = data[:,:,inds_data]
    labels = labels[:,:,inds_labels]

    return data, labels


def normalize(data):
    dim = data.shape[-1]
    return np.divide(data-cfg.SHIFT[:dim], cfg.NORMALIZE[:dim])


def unnormalize(data):
    dim = data.shape[-1]
    return np.multiply(data, cfg.NORMALIZE[:dim]) + cfg.SHIFT[:dim]