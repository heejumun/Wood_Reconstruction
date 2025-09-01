import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *
from utils import registry
import pandas as pd

@DATASETS.register_module()
class PUGAN_TREE(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            
            model_id = line.split('.')[0]
            self.file_list.append({
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='TreeVX')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.nanmean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.nansum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        name = sample['file_path'].split('.')[0]

        data = IO.get(os.path.join(self.pc_path, sample['file_path']))
        data = data.replace({'\ufeff': ''}, regex=True).values
        data = data.astype(np.float32)
        data = self.pc_norm(data)

        return sample['model_id'], data
    
    def __len__(self):
        return len(self.file_list)
