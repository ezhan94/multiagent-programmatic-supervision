import torch
from torch.utils.data import Dataset
from importlib import import_module


class GeneralDataset(Dataset):

    def __init__(self, dataset_name, train=True, normalize_data=True, subsample=1):
        dataset = import_module('datasets.{}'.format(dataset_name))

        data, labels = dataset.fetch(train)
        data, labels = dataset.preprocess(data, labels)
        data, labels = data[:,::subsample,:], labels[:,::subsample,:]

        if normalize_data:
            data = dataset.normalize(data)

        self.train = train
        self.data = torch.Tensor(data) # torch.FloatTensor
        self.labels = torch.Tensor(labels).long()
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
        