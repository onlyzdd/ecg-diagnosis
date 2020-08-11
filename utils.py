import numpy as np

from sklearn.metrics import roc_auc_score
import wfdb


def find_optimal_threshold_GBeta(y_true, y_prob):
    thresholds = np.linspace(0, 1, 100)


def prepare_input(ecg_file: str):
    if ecg_file.endswith('.mat'):
        ecg_file = ecg_file[:-4]
    ecg_data, _ = wfdb.rdsamp(ecg_file)
    nsteps, nleads = ecg_data.shape
    ecg_data = ecg_data[-5000, :]
    result = np.zeros((5000, nleads))
    result[-nsteps:, :] = ecg_data
    return result.T


def cal_metric(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    return auc
