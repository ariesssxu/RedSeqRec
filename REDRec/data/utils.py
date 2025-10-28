import importlib
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

def bulid_dataloader(config, local_rank=None, world_size=None):
    '''
    split dataset, generate user history sequence, train/valid/test dataset
    '''
    dataset_dict = {
        'REDRec': ('REDRecDataset'),
    }
    
    model_name = config.model.model_name
    dataset_module = importlib.import_module('REDRec.data.dataset')
    train_set_name = dataset_dict[model_name]

    train_set_class = getattr(dataset_module, train_set_name)
    
    train_dataset = train_set_class(local_rank, world_size, config)
    train_num_workers = config.data.train_num_workers
    # train_loader = DataLoader(train_dataset, batch_size=None, batch_sampler=None, num_workers=train_num_workers, shuffle=False, prefetch_factor=2)
    train_loader = DataLoader(train_dataset, batch_size=None, batch_sampler=None, num_workers=train_num_workers, shuffle=False)
    
    return train_loader, train_loader, train_loader