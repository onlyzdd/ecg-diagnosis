import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

import wfdb

import os


class ECGDataset(Dataset):
    def __init__(self, data_dir, label_csv, folds, leads):
        super(ECGDataset, self).__init__()
        df = pd.read_csv(label_csv)
        df = df[df['fold'].isin(folds)]
        self.data_dir = data_dir
        self.reference = df
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if leads == 'all':
            self.use_leads = np.where(np.in1d(self.leads, self.leads))[0]
        else:
            self.use_leads = np.where(np.in1d(self.leads, leads))[0]
        self.nleads = len(self.use_leads)
        self.classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
        self.n_classes = len(self.classes)
        self.label_dict = {}

    def __getitem__(self, index: int):
        row = self.reference.iloc[index]
        patient_id = row['patient_id']
        ecg_data, _ = wfdb.rdsamp(os.path.join(self.data_dir, patient_id))
        nsteps, _ = ecg_data.shape
        ecg_data = ecg_data[-5000:, self.use_leads]
        result = np.zeros((5000, self.nleads)) # 10 s, 500 Hz
        result[-nsteps:, :] = ecg_data
        if self.label_dict.get(patient_id):
            labels = self.label_dict.get(patient_id)
        else:
            labels = row[self.classes].to_numpy(dtype=np.float32)
            self.label_dict[patient_id] = labels
        return torch.from_numpy(result.T).float(), torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.reference)
