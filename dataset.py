import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import TypeVar

import wfdb

import os
import ast
import json

T_co = TypeVar('T_co', covariant=True)

class ECGDataset(Dataset):
    def __init__(self, data_dir, label_csv, folds) -> None:
        super(ECGDataset, self).__init__()
        df = pd.read_csv(label_csv)
        df = df[df['fold'].isin(folds)]
        self.data_dir = data_dir
        self.reference = df
        self.classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
        self.n_classes = len(self.classes)
        self.label_dict = {}

    def __getitem__(self, index: int) -> T_co:
        row = self.reference.iloc[index]
        patient_id = row['patient_id']
        ecg_data, _ = wfdb.rdsamp(os.path.join(self.data_dir, patient_id))
        nsteps, nleads = ecg_data.shape
        ecg_data = ecg_data[-10000:, :]
        result = np.zeros((10000, nleads)) # 10 s, 500 Hz
        result[-nsteps:, :] = ecg_data
        if self.label_dict.get(patient_id):
            labels = self.label_dict.get(patient_id)
        else:
            labels = row[self.classes].to_numpy(dtype=np.float32)
            self.label_dict[patient_id] = labels
        return torch.from_numpy(result.T).float(), torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.reference)
