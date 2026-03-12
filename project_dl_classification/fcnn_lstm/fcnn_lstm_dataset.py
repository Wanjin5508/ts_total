import random

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import RobustScaler
import joblib


class PartialDischargeDataset(Dataset):
    def __init__(self, dataframe, test_stage=False):
        """
        """
        self.dataframe = dataframe
        self.test_stage = test_stage

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get sample from dataframe at index idx
        sample = self.dataframe.iloc[idx]  # timedomain signal

        # Extract features and target
        # input = torch.tensor((sample['tds']/np.amax(sample['tds']))[:4], dtype=torch.float32)
        normalized = self._normalize(sample['raw_signal'])
        input = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
        # target = torch.tensor(0 if sample['cluster_id'] == -1 else 1, dtype=torch.long)
        target = torch.tensor(sample['label'], dtype=torch.long)  # if not self.test_stage else None
        return input, target

    def _normalize(self, arr: np.ndarray):
        max = np.max(arr)
        min = np.min(arr)
        normalized = (arr - min) / (max - min + 1e-8)
        return normalized
    
    def get_sample_for_id(self, id):
        condition = self.dataframe['label'] == id
        indices = self.dataframe.index[condition]

        if len(indices) == 0:
            raise ValueError("No samples found for the given id")

        self.last_rdm_idx = random.choice(indices)
        inp, label = self.__getitem__(self.last_rdm_idx)
        print(f'get sample for id {id}, label = {label}')
        return self.__getitem__(self.last_rdm_idx)


    def get_false_positive(self, pred):
        if pred == 0:
            return
        condition = self.dataframe['label'] == 0
        indices = self.dataframe.index[condition]

        if len(indices) == 0:
            raise ValueError("No false positive samples")
        inp, label = self.__getitem__(self.last_rdm_idx)
        print(f'get sample for id {id}, label = {label}')
        return self.__getitem__(self.last_rdm_idx)


    def scale_raw_signal(self):
        # 已弃用，因为这种方式需要在训练阶段保存scaler，测试阶段加载scaler，无法保证训练数据和测试数据的分布一致性
        if not self.test_stage:
            scaler = RobustScaler()
            self.dataframe['scaled_signal'] = self.dataframe['raw_signal'].apply(lambda input: scaler.fit_transform(input.reshape(-1,1)) )
            self.scaler = scaler
            joblib.dump(scaler, self.scaler_name)
        else:
            # test stage
            print("InTest stage| scale_raw_signal is deactivated, make sure that you have used the static method\n")


def custom_collate(batch):  # handle padding within batches
    # print(f'custom collate batch shape: {len(batch)}')
    inputs, labels = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    return padded_inputs.view(padded_inputs.shape[0], 1, -1), torch.tensor(labels)  # view to keep get extra dim for model input


def custom_collate_with_lengths(batch):
    """for RNN,
    batch: list of tuples (x, y) = List[(tensor(C, L_i), int)]
    labels(N,)

    return:
        xs: (N, C, L_max),
        ys: (N, )
        Lengths: (N, )
    """
    xs, ys, lengths = [], [], []
    for x, y, in batch:
        # print(f"custom_collate_with_lengths: x: {x.size()}, y: {y.size()}")
        assert x.ndim == 2 and x.shape[0] >= 1, "except (C, L)"
        C, Li = x.shape
        xs.append(x.transpose(0, 1))  # (L_i, C)
        ys.append(torch.as_tensor(y, dtype=torch.long))
        lengths.append(Li)

    xs = pad_sequence(xs, batch_first=True, padding_value=0.0)  # (N, L_max, C)
    xs = xs.transpose(1, 2).contiguous()  # -> (N, C, L_max)
    ys = torch.stack(ys, dim=0)
    lengths = torch.as_tensor(lengths, dtype=torch.long)
    return xs, ys, lengths


