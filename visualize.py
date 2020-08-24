import os
import argparse
from glob import glob

import torch
import numpy as np
import matplotlib.pyplot as plt

from resnet import resnet34
from utils import prepare_input


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='data/CPSC', help='Data directory')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use GPU')
    return parser.parse_args()


def plot_ecg(data, leads, patient_id, label):
    """plot 1 column ecg"""
    n_cols = 1
    n_rows = len(leads) // n_cols
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 16))
    for i in range(n_rows):
        axs[i].plot(data[i], linewidth=0.4)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_ylabel(leads[i])
        yabs_max = abs(max(axs[i].get_ylim(), key=abs))
        axs[i].set_ylim(ymin=-yabs_max, ymax=yabs_max)
    plt.savefig(f'imgs/{patient_id}.png')
    plt.close(fig)


def plot_ecg2(data, leads, patient_id, label):
    """plot 2 column ecg"""
    n_cols=2
    n_rows = len(leads) // n_cols
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(9, 9))
    for j in range(n_rows):
        for i in range(n_cols):
            axs[j, i].plot(data[i * n_rows + j], linewidth=0.4)
            axs[j, i].spines['top'].set_visible(False)
            axs[j, i].spines['right'].set_visible(False)
            axs[j, i].set_xticks([])
            axs[j, i].set_yticks([])
            axs[j, i].set_ylabel(leads[i * n_rows + j])
            yabs_max = abs(max(axs[j, i].get_ylim(), key=abs))
            axs[j, i].set_ylim(ymin=-yabs_max, ymax=yabs_max)
    plt.savefig(f'imgs/{patient_id}-c2.png')
    plt.close(fig)


def plot_shap(ecg_data, sv_data, leads, patient_id, label):
    """plot ecg with shap values"""
    nleads = len(leads)
    x = range(ecg_data.shape[1])
    plt.figure(figsize=(20, nleads * 2))
    middle = 0.0001
    fig, axs = plt.subplots(nleads)
    fig.suptitle(label)
    for i in range(nleads):
        sv_upper = np.ma.masked_where(sv_data[i] >= middle, ecg_data[i])
        sv_lower = np.ma.masked_where(sv_data[i] < middle, ecg_data[i])
        axs[i].plot(x, sv_upper, x, sv_lower)
    plt.savefig(f'results/{patient_id}.png')
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    sv = np.load('results/A001.npy')
    model = resnet34(input_channels=12).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    nsamples = 10000 # last 10 seconds
    llabel = ['NORM', 'AFIB', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    lleads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    inputpaths = glob(os.path.join(args.input_dir, 'A001*.mat'))
    inputs = torch.stack([torch.from_numpy(prepare_input(input0)).float() for input0 in inputpaths])
    results = model(inputs).detach().numpy()

    label_idx = np.argmax(results, axis=1)
    labels = [llabel[i] for i in label_idx]
    for idx in range(len(inputs)):
        input0 = inputpaths[idx]
        patient_id = input0[-9:-4]
        label = labels[idx]
        ecg_data = inputs[idx].numpy()[-nsamples:, :]
        sv_data = sv[label_idx[idx]][idx][-nsamples:, :].T
        plot_shap(ecg_data, sv_data, lleads, patient_id, label)
