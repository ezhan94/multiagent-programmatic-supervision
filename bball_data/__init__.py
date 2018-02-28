import numpy as np
import os
import pickle

import torch
from torch.utils.data import Dataset

import bball_data.cfg as cfg
from bball_data.utils import normalize


DATAPATH = cfg.DATAPATH
N_TRAIN = cfg.N_TRAIN
N_TEST = cfg.N_TEST
SEQ_LENGTH = cfg.SEQUENCE_LENGTH
X_DIM = cfg.SEQUENCE_DIMENSION


class BBallData(Dataset):

    train_filename = 'Xtr_role'
    test_filename = 'Xte_role'


    def __init__(self, train=True, preprocess=True, subsample=1):
        self.preprocess = preprocess
        self.train = train
        self.n = N_TRAIN if train else N_TEST
        self.subsample = subsample

        if self.train:
            self.train_data, self.train_labels = self.fetch_data(self.train_filename)
        else:
            self.test_data, self.test_labels = self.fetch_data(self.test_filename)

    
    def __getitem__(self, index):
        if self.train:
            seq, target = self.train_data[index], self.train_labels[index]
        else:
            seq, target = self.test_data[index], self.test_labels[index]

        return seq.unsqueeze(0), target


    def __len__(self):
        return self.n


    def fetch_data(self, filename):
        labels = pickle.load(open(DATAPATH+filename+'_macro.p', 'rb'))
        data = np.zeros((self.n, SEQ_LENGTH, X_DIM))

        if os.path.isfile(DATAPATH+filename+'.p'):
            data = pickle.load(open(DATAPATH+filename+'.p', 'rb'))
        else:
            counter = 0
            file = open(DATAPATH+file+'.txt')
            for line in file:
                t = counter % SEQ_LENGTH
                s = int((counter - t) / SEQ_LENGTH)
                data[s][t] = line.strip().split(' ')
                counter += 1
            pickle.dump(data, open(DATAPATH+filename+'.p', 'wb'))

        # get just the offensive players
        inds_data = cfg.COORDS['offense']['xy']
        inds_labels = [int(i/2) for i in inds_data[::2]]

        # subsample data
        data = data[:,::self.subsample,inds_data]
        labels = labels[:,::self.subsample,inds_labels]

        if self.preprocess:
            data = normalize(data)

        # convert to torch tensors
        data = torch.Tensor(data) # torch.FloatTensor
        labels = torch.Tensor(labels).long()

        return data, labels