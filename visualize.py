import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
import wfdb
from tqdm import tqdm


def plot_ecg(leads, data, title):
    n_cols = 1
    n_rows = len(leads) // n_cols
    f, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 16))
    for i in range(n_rows):
        axs[i].plot(data[i])
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_ylabel(leads[i])
        yabs_max = abs(max(axs[i].get_ylim(), key=abs))
        axs[i].set_ylim(ymin=-yabs_max, ymax=yabs_max)
    plt.savefig(f'imgs/{title}.png')
    plt.close(f)


def plot_ecg2(leads, data, title):
    n_cols=2
    n_rows = len(leads) // n_cols
    f, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 8))
    for j in range(n_rows):
        for i in range(n_cols):
            axs[j, i].plot(data[i * n_rows + j])
            axs[j, i].spines['top'].set_visible(False)
            axs[j, i].spines['right'].set_visible(False)
            axs[j, i].set_xticks([])
            axs[j, i].set_yticks([])
            axs[j, i].set_ylabel(leads[i * n_rows + j])
            yabs_max = abs(max(axs[j, i].get_ylim(), key=abs))
            axs[j, i].set_ylim(ymin=-yabs_max, ymax=yabs_max)
    plt.savefig(f'imgs/{title}-c2.png')
    plt.close(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-paths', type=str, default='data/CPSC/A0010.mat', help='Path to .mat file')
    args = parser.parse_args()
    recordpaths = glob(args.record_paths)
    for mat_file in tqdm(recordpaths):
        if mat_file.endswith('.mat') or mat_file.endswith('.hea'):
            mat_file = mat_file[:-4]
        patient_id = os.path.basename(mat_file)
        ecg_data, meta_data = wfdb.rdsamp(mat_file)
        leads = meta_data['sig_name']
        plot_ecg(leads=leads, data=ecg_data.T, title=os.path.basename(mat_file))
        plot_ecg2(leads=leads, data=ecg_data.T, title=os.path.basename(mat_file))
