import argparse
import os

import torch
import wfdb
from resnet import resnet34

import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/CPSC', help='Directory for data dir')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--model-path', type=str, default='models/resnet34.pth', help='Path to saved model')
    return parser.parse_args()

args = parse_args()


if __name__ == "__main__":
    data_dir = args.data_dir
    database = os.path.basename(data_dir)
    label_csv = os.path.join(data_dir, 'labels.csv')
    batch_size = args.batch_size
    df = pd.read_csv(label_csv)
    patient_ids = df[df['fold'] == 9]['patient_id']
    net = resnet34(input_channels=12)
    net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    net.eval()
    for i in range(0, len(patient_ids), batch_size):
        j = (i + batch_size) if i + batch_size < len(patient_ids) else len(patient_ids)
        batch_ids = patient_ids[i:j]
        ecg_data_array = []
        for patient_id in batch_ids:
            ecg_data = wfdb.rdsamp(os.path.join(data_dir, patient_id))[0]
            nsteps, nleads = ecg_data.shape
            ecg_data = ecg_data[-5000:, :]
            result = np.zeros((5000, nleads)) # 10 s, 500 Hz
            result[-nsteps:, :] = result
            ecg_data_array.append(result.T)
        output = net(torch.from_numpy(np.asarray(ecg_data_array)).float())
