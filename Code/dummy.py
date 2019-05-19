# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 12:28:37 2018

@author: Harshit
"""
import torch
from torch.utils.data.dataset import Dataset

class Dataset_load(Dataset):
    def __init__(self, path, chunksize, nb_samples):
        self.path = path
        self.chunksize = chunksize
        self.len = int(nb_samples / self.chunksize)
    def __getitem__(self, index):
        return index
    def __len__(self):
        return self.len
    
train_dataset = Dataset_load("TrainData.ctf", chunksize=1, nb_samples=450000)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=250, num_workers=4, shuffle=False)

for i, index in enumerate(train_loader):
    print(index)