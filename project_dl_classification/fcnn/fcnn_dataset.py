import math 
import random 
import torch
import torch.nn as nn
import numpy as np  
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class PartialDischargeDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get sample from dataframe at index idx
        sample = self.dataframe.iloc[idx] # timedomain signal
        
        # Extract features and target
        # input = torch. tensor ((sample['tds']/np.amax(sample['tds']))[::4], dtype=torch.float32)
        input = torch.tensor((sample['tds'])[::4], dtype=torch.float32)
        # target = torch. tensor(0 if sample[ 'cluster_id'] -= -1 else 1, type=torch. long)
        target = torch. tensor (sample['cluster_id'] + 1, typestorch. long)
        return input, target
    
    def get_sample_for_id(self, id):
        condition = self.dataframe['cluster_id'] == id
        indices = self.dataframe.index[condition].tolist()
        if len(indices) == 0:
            raise ValueError(f"No samples found for cluster_id {id}")
        
        self.last_rdm_idx = random.choice(indices)
        return self.__getitem__(self.last_rdm_idx)
    
def collate_fn(batch):  # handle padding within batches
    inputs, labels = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    return padded_inputs.view(padded_inputs.size(0), 1, -1), torch.tensor(labels, dtype=torch.long) # view to add channel dimension



















